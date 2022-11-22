import os
import json
import logging
from datetime import datetime
from argparse import ArgumentParser

import torch
import numpy as np
import transformers
from tqdm import tqdm
from attrdict import AttrDict
from datasets import load_dataset
from transformers.data.processors.squad import SquadResult
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate

from evaluate_squad import official_evaluate
from corpus import load_examples, load_features_and_dataset
from utils_qa import TOKENIZER_CLASSES, MODEL_FOR_QUESTION_ANSWERING, TqdmLoggingHandler

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()


def create_logger(output_dir: str):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)


def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    # Logger 설정
    create_logger(args.output_dir)

    # Tokenizer 생성
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )

    # Raw Datasets 생성
    data_files = {
        'train': [os.path.join('train', file_name) for file_name in os.listdir(args.train_dir)],
        'validation': [os.path.join('validation', file_name) for file_name in os.listdir(args.validation_dir)],
    }
    raw_datasets = load_dataset('json', data_dir='data', data_files=data_files, field='data')

    # Loading train dataset
    train_examples = load_examples(raw_datasets['train'], evaluate=False)
    train_dataset, _ = load_features_and_dataset(args, train_examples, tokenizer, evaluate=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # Loading evaluate dataset
    eval_examples = load_examples(raw_datasets['validation'], evaluate=True)
    eval_dataset, eval_features = load_features_and_dataset(args, eval_examples, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # model 생성
    model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(args.model_name_or_path)
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(args.device)

    # Optimizer setting
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # LR_Scheduler setting
    total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Train Start
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_steps)

    official_f1 = None
    loss_list_between_log_interval = []
    for epoch in range(args.num_train_epochs):
        # Step(batch) 루프
        for step, batch in tqdm(enumerate(train_dataloader), f"[TRAIN] EP:{epoch}", total=len(train_dataloader)):
            global_step = len(train_dataloader) * epoch + step + 1
            optimizer.zero_grad()

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            loss_list_between_log_interval.append(loss.item())
            if global_step % args.logging_steps == 0:
                mean_loss = np.mean(loss_list_between_log_interval)
                logger.info(
                    f"EP:{epoch} global_step:{global_step} "
                    f"loss:{mean_loss:.4f}"
                )
                loss_list_between_log_interval.clear()

        # -- Evaluate & Save model result -- #
        eval_result = evaluate(args, model, tokenizer, eval_dataloader, eval_examples, eval_features, epoch)
        for key in sorted(eval_result.keys()):
            logger.info("  %s = %s", key, str(eval_result[key]))

        if official_f1 is None or official_f1 < eval_result['official_f1']:
            official_f1 = eval_result['official_f1']
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)


def evaluate(args, model, tokenizer, dataloader, examples, features, epoch=None):
    # Evaluation
    logger.info("***** Running evaluation {} *****".format(epoch))
    logger.info("  Num examples = %d", len(dataloader.dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()  # 모델의 AutoGradient 연산을 비활성화하고 평가 연산 모드로 설정 (메모리 사용 및 연산 효율화를 위해)

    all_results = []
    eval_iterator = tqdm(dataloader)
    for step, batch in enumerate(eval_iterator):
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            outputs = model(**inputs)

        example_indices = batch[3]
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            start_logits = outputs.start_logits[i].detach().cpu().tolist()
            end_logits = outputs.end_logits[i].detach().cpu().tolist()
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(epoch))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(epoch))

    output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    # Write the result
    # Write the evaluation result on file
    output_dir = os.path.join(args.output_dir, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(
        output_dir,
        "eval_result_{}_{}.txt".format(
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            epoch
        )
    )

    with open(output_eval_file, "w", encoding='utf-8') as f:
        official_eval_results = official_evaluate(examples, predictions)
        results.update(official_eval_results)

    return results

if __name__ == '__main__':
    cli_parser = ArgumentParser()

    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default="koelectra_base_v3.json")

    cli_args = cli_parser.parse_args()

    main(cli_args)

from tqdm import tqdm
from transformers import SquadExample, squad_convert_examples_to_features


def load_examples(raw_datasets, evaluate=False):
	examples = []
	for tensor_dict in tqdm(raw_datasets):
		if not evaluate:
			answer = tensor_dict["answers"]["text"][0]
			answer_start = tensor_dict["answers"]["answer_start"][0]
			answers = []
		else:
			answers = [
				{
					"answer_start": tensor_dict["answers"]["answer_start"][0],
					"text": tensor_dict["answers"]["text"][0]
				}
			]

			answer = None
			answer_start = None

		example = SquadExample(
			qas_id=tensor_dict["id"],
			question_text=tensor_dict["question"],
			context_text=tensor_dict["context"],
			answer_text=answer,
			start_position_character=answer_start,
			title=tensor_dict["title"],
			answers=answers,
		)

		examples.append(example)

	return examples

def load_features_and_dataset(args, examples, tokenizer, evaluate=False):
	# Prepare dataset
	features, dataset = squad_convert_examples_to_features(
		examples=examples,
		tokenizer=tokenizer,
		max_seq_length=args.max_seq_length,
		doc_stride=args.doc_stride,
		max_query_length=args.max_query_length,
		is_training=not evaluate,
		return_dataset="pt",
		threads=args.threads,
	)

	return dataset, features

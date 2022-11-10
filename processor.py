from datasets import load_dataset
from transformers import AutoTokenizer


class Preprocessor:
    def __init__(self, tokenizer):
        # self.args = args
        self.max_length = 384
        self.stride = 100
        self.tokenizer = tokenizer

    def preprocess_training_examples(self, examples):
        # 주어진 질문과 Context 를 Tokenizing 수행
        # 이때 질문과 Context 의 길이가 max_seq_length 를 넘으면 stride 만큼 슬라이딩 윈도우를 수행함.
        # 즉, 하나의 example 에서 일부분이 겹치는 여러 sequence가 생길 수 있음.
        questions = [q.strip() for q in examples["question"]]
        tokenized_examples = self.tokenizer(
            questions,
            examples["context"],
            max_length = self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True, # 길이가 넘어가는 토큰들을 반환할 것인지 여부를 결정
            return_offsets_mapping=True, # 각 토큰에 대해 (char_start, char_end) 정보를 반환할 것인지 여부를 결정
            padding="max_length",
        )

        # example 하나가 여러 sequence 대응하는 경우를 위해 매핑이 필요함.
        # offset_mappings으로 토큰이 원본 context 내 몇 번째 글자부터 몇 번째 글자까지 해당하는지 확인할 수 있음.
        offset_mapping = tokenized_examples.pop("offset_mapping")
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # 정답지를 만들기 위한 리스트
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # 해당 example에 해당하는 sequence를 찾음.
            sequence_ids = tokenized_examples.sequence_ids(i)

            # sequence가 속하는 example을 찾음.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # 만약 주어진 정답이 없을 경우 cls index를 정답 위치로 설정
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Context에서 정답의 시작점과 끝점
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Context의 시작 위치를 찾는다.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # context의 마지막 위치를 찾는다.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # 만일 정답이 context에 완전히 포함되지 않았다면 레이블을 (cls_index, cls_index)로 설정
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index와 token_end_index를 answer의 시작점과 끝점으로 변경
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def preprocess_validation_examples(self, examples):
        # 주어진 질문과 Context 를 Tokenizing 수행
        # 이때 질문과 Context 의 길이가 max_seq_length 를 넘으면 stride 만큼 슬라이딩 윈도우를 수행함.
        # 즉, 하나의 example 에서 일부분이 겹치는 여러 sequence가 생길 수 있음.
        questions = [q.strip() for q in examples["question"]]
        tokenized_examples = self.tokenizer(
            questions,
            examples["context"],
            max_length = self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True, # 길이가 넘어가는 토큰들을 반환할 것인지 여부를 결정
            return_offsets_mapping=True, # 각 토큰에 대해 (char_start, char_end) 정보를 반환할 것인지 여부를 결정
            padding="max_length",
        )

        # example 하나가 여러 sequence 대응하는 경우를 위해 매핑이 필요함.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # sequence가 대응하는 example id를 저장하는 리스트
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # 해당 example에 해당하는 sequence를 찾음.
            sequence_ids = tokenized_examples.sequence_ids(i)

            # sequence가 속하는 example을 찾음.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 context에 속하는지 여부를 쉽게 판별할 수 있음
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

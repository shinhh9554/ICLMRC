class Preprocessor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def preprocess_training(self, examples):
        questions = [q.strip() for q in examples["question"]]
        tokenized_examples = self.tokenizer(
            questions,
            examples["context"],
            max_length = self.args.max_length,
            truncation="only_second",
            stride=self.args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = tokenized_examples.pop("offset_mapping")
        sample_map = tokenized_examples.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []


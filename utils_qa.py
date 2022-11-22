import tqdm
import logging

from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    ElectraForQuestionAnswering,
)

CONFIG_CLASSES = {
    "koelectra-base-v3": ElectraConfig,
    "koelectra-small-v3": ElectraConfig,
}

TOKENIZER_CLASSES = {
    "koelectra-base-v3": ElectraTokenizer,
    "koelectra-small-v3": ElectraTokenizer,
}

MODEL_FOR_QUESTION_ANSWERING = {
    "koelectra-base-v3": ElectraForQuestionAnswering,
    "koelectra-small-v3": ElectraForQuestionAnswering,
}

"""
Using TQDM with logging. Code from under link!
https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
"""

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
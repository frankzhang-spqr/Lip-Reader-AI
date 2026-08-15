"""SentencePiece subword tokenizer for open-vocabulary English lip-reading."""

import os

import sentencepiece as spm


def train_tokenizer(text_files: list[str], model_prefix: str, vocab_size: int = 5000) -> None:
    """Train a unigram SentencePiece model on the given transcript files.

    If the requested vocab_size exceeds what the corpus supports (e.g. a small
    custom dataset), the size is reduced automatically.
    """
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    prefix = model_prefix[: -len(".model")] if model_prefix.endswith(".model") else model_prefix
    size = vocab_size
    while size >= 16:
        try:
            spm.SentencePieceTrainer.train(
                input=",".join(text_files),
                model_prefix=prefix,
                vocab_size=size,
                model_type="unigram",
                character_coverage=1.0,
            )
            if size != vocab_size:
                print(f"  note: vocab reduced to {size} to fit the corpus")
            return
        except RuntimeError as exc:
            if "Vocabulary size too high" not in str(exc):
                raise
            size //= 2
    raise RuntimeError("could not fit a SentencePiece vocabulary to the corpus")


class Tokenizer:
    """Wraps a SentencePiece model with the AutoAVSR-style token list.

    token_list = ['<blank>'] + subwords + ['<eos>']
    blank = 0, eos = len(token_list) - 1.
    """

    def __init__(self, model_file: str, vocab_size: int = 5000):
        assert os.path.isfile(model_file), f"missing tokenizer model: {model_file}"
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.n_pieces = self.sp.get_piece_size()
        self.vocab_size = vocab_size
        self.token_list = ["<blank>"] + [self.sp.id_to_piece(i) for i in range(self.n_pieces)] + ["<eos>"]
        self.char2id = {t: i for i, t in enumerate(self.token_list)}
        self.blank = 0
        self.eos = len(self.token_list) - 1

    def text_to_ids(self, text: str) -> list[int]:
        pieces = self.sp.encode(text, out_type=str)
        return [self.char2id[p] for p in pieces]

    def ids_to_text(self, ids: list[int]) -> str:
        chars = [self.token_list[i] for i in ids if 0 < i < self.eos]
        return "".join(chars).replace("▁", " ").strip()

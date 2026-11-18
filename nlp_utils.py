# nlp_utils.py - Text preprocessing + vocabulary shared by training and serving.
# Pure Python (no NLTK downloads needed), so the same code runs everywhere.
import re

_TOKEN_RE = re.compile(r"[a-z]+")


def _normalize(word):
    """Very light plural stripping: 'courses' -> 'course', keeps 'class', 'this'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text):
    """Lowercase, drop apostrophes (what's -> whats), keep letters only."""
    text = text.lower().replace("'", "").replace("\u2019", "")
    return [_normalize(w) for w in _TOKEN_RE.findall(text)]


class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"
    PAD_ID, UNK_ID = 0, 1

    def __init__(self, word2idx=None):
        self.word2idx = word2idx or {self.PAD: 0, self.UNK: 1}

    def build(self, token_lists):
        for tokens in token_lists:
            for w in tokens:
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.word2idx)

    def encode(self, tokens, max_len):
        ids = [self.word2idx.get(w, self.UNK_ID) for w in tokens][:max_len]
        if not ids:                      # empty / non-alphabetic input
            ids = [self.UNK_ID]
        return ids + [self.PAD_ID] * (max_len - len(ids))

    def __len__(self):
        return len(self.word2idx)

# model.py - BiLSTM + attention intent classifier (PyTorch)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMIntentClassifier(nn.Module):
    """Embedding -> BiLSTM -> masked attention pooling -> dropout -> Linear."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        mask = x != 0                                     # (B, L)
        lengths = mask.sum(1).clamp(min=1).cpu()
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        scores = self.attention(out).squeeze(-1)          # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (weights * out).sum(dim=1)
        return self.fc(self.dropout(context))

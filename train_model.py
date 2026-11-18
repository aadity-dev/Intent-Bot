# train_model.py - Deep learning intent classifier (PyTorch BiLSTM + attention)
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from model import LSTMIntentClassifier
from nlp_utils import Vocabulary, tokenize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

CONFIG = dict(embed_dim=64, hidden_dim=96, num_layers=1, dropout=0.3, max_len=16)
EPOCHS = 120
BATCH_SIZE = 16
LR = 3e-3
WORD_DROPOUT = 0.12      # randomly turn known words into <UNK> while training
DEFAULT_SAMPLES = 14     # synthetic "unknown input" examples for the default class


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_intents(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["intents"]


def build_dataset(intents):
    classes, sentences, labels = [], [], []
    for it in intents:
        classes.append(it["tag"])
        for p in it["patterns"]:
            if p.strip():
                sentences.append(tokenize(p))
                labels.append(it["tag"])
    return classes, sentences, labels


def stratified_split(labels, test_ratio=0.2):
    by_class = {}
    for i, lbl in enumerate(labels):
        by_class.setdefault(lbl, []).append(i)
    train, test = [], []
    rng = random.Random(SEED)
    for idxs in by_class.values():
        rng.shuffle(idxs)
        n_test = int(round(len(idxs) * test_ratio)) if len(idxs) >= 5 else 0
        test += idxs[:n_test]
        train += idxs[n_test:]
    return train, test


def fit(vocab, classes, sentences, labels, device, epochs=EPOCHS, verbose=True):
    """Train a fresh model on the given examples and return it."""
    cls_idx = {c: i for i, c in enumerate(classes)}
    max_len = CONFIG["max_len"]

    X = [vocab.encode(s, max_len) for s in sentences]
    y = [cls_idx[l] for l in labels]
    # Synthetic samples so gibberish / out-of-domain text lands on "default"
    if "default" in cls_idx:
        for n in range(DEFAULT_SAMPLES):
            X.append(vocab.encode(["<unk>"] * (1 + n % 4), max_len))
            y.append(cls_idx["default"])

    X = torch.tensor(X, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    model = LSTMIntentClassifier(len(vocab), num_classes=len(classes), **{
        k: v for k, v in CONFIG.items() if k != "max_len"}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X), device=device)
        total, correct, loss_sum = 0, 0, 0.0
        for start in range(0, len(X), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb, yb = X[idx].clone(), y[idx]
            drop = (torch.rand_like(xb, dtype=torch.float) < WORD_DROPOUT) & (xb > 1)
            xb[drop] = vocab.UNK_ID
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        sched.step()
        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"  epoch {epoch:>3}  loss {loss_sum / total:.4f}  train acc {correct / total:.1%}")
    return model


@torch.no_grad()
def predict(model, vocab, classes, sentences, device):
    model.eval()
    X = torch.tensor([vocab.encode(s, CONFIG["max_len"]) for s in sentences],
                     dtype=torch.long, device=device)
    probs = torch.softmax(model(X), dim=1)
    conf, idx = probs.max(1)
    return [classes[i] for i in idx.tolist()], conf.tolist()


def main():
    print("Intent chatbot - deep learning training (PyTorch BiLSTM + attention)")
    print("=" * 66)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    intents = load_intents(os.path.join(BASE_DIR, "intents.json"))
    classes, sentences, labels = build_dataset(intents)
    print(f"{len(sentences)} training patterns, {len(classes)} intents")

    # 1) Honest evaluation: train on 80%, test on the held-out 20%
    tr, te = stratified_split(labels)
    vocab_eval = Vocabulary()
    vocab_eval.build([sentences[i] for i in tr])
    print(f"\n[1/2] Held-out evaluation ({len(tr)} train / {len(te)} test)")
    model = fit(vocab_eval, classes, [sentences[i] for i in tr], [labels[i] for i in tr], device)
    preds, _ = predict(model, vocab_eval, classes, [sentences[i] for i in te], device)
    truth = [labels[i] for i in te]
    acc = sum(p == t for p, t in zip(preds, truth)) / max(len(te), 1)
    print(f"\nHeld-out accuracy: {acc:.1%}  ({sum(p == t for p, t in zip(preds, truth))}/{len(te)})")
    wrong = [(" ".join(sentences[i]), t, p) for i, t, p in zip(te, truth, preds) if t != p]
    for text, t, p in wrong:
        print(f"   miss: '{text}'  expected={t}  got={p}")

    # 2) Final model: train on ALL patterns and save
    print(f"\n[2/2] Training final model on all {len(sentences)} patterns")
    set_seed()
    vocab = Vocabulary()
    vocab.build(sentences)
    print(f"Vocabulary size: {len(vocab)}")
    model = fit(vocab, classes, sentences, labels, device)
    preds, _ = predict(model, vocab, classes, sentences, device)
    train_acc = sum(p == t for p, t in zip(preds, labels)) / len(labels)
    print(f"Final training accuracy: {train_acc:.1%}")

    out_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": CONFIG,
        "word2idx": vocab.word2idx,
        "classes": classes,
        "heldout_accuracy": float(acc),
    }, os.path.join(out_dir, "dl_model.pt"))
    print(f"Saved model to {out_dir}/dl_model.pt")

    print("\nQuick check:")
    tests = ["Hello there!", "Thanks for your help", "What's the weather like?",
             "Tell me about machine learning", "My code is not working",
             "I'm feeling demotivated", "asdf qwerty zxcv", "purple monkey dishwasher"]
    preds, confs = predict(model, vocab, classes, [tokenize(t) for t in tests], device)
    for t, p, c in zip(tests, preds, confs):
        print(f"  {t!r:<38} -> {p:<15} {c:.0%}")


if __name__ == "__main__":
    main()

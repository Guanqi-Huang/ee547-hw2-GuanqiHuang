import argparse
import json
import os
import re
import time
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WORD_RE = re.compile(r"[a-z]+", re.I)
def tokenize(s: str):
    return WORD_RE.findall(s or "")

def load_abstracts(papers_json: str):
    with open(papers_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "papers" in data:
        data = data["papers"]
    abstracts = [ (p.get("arxiv_id") or p.get("id") or ""), p.get("abstract","") ][1]
    return [p.get("abstract","") for p in data if isinstance(p, dict)]

def build_vocab(abstracts, vocab_size: int):
    counter = Counter()
    total_words = 0
    for abs_ in abstracts:
        toks = tokenize(abs_)
        counter.update(toks)
        total_words += len(toks)
    # keep the top vocabulary size words
    vocab = [w for w, _ in counter.most_common(vocab_size)]
    w2i = {w:i for i, w in enumerate(vocab)}
    return w2i, vocab, total_words

def vectorize(abstracts, w2i, binary=True):
    V = len(w2i)
    X = torch.zeros((len(abstracts), V), dtype=torch.float32)
    for i, abs_ in enumerate(abstracts):
        for w in tokenize(abs_):
            j = w2i.get(w)
            if j is not None:
                if binary:
                    X[i, j] = 1.0
                else:
                    X[i, j] += 1.0
    return X

# Model
class BoWAutoencoder(nn.Module):
    def __init__(self, V:int, H:int, E:int):
        super().__init__()
        self.enc1 = nn.Linear(V, H)
        self.enc2 = nn.Linear(H, E)
        self.dec1 = nn.Linear(E, H)
        self.dec2 = nn.Linear(H, V)
        self.act = nn.ReLU()

    def forward(self, x):
        h1 = self.act(self.enc1(x))
        z  = self.act(self.enc2(h1))
        h2 = self.act(self.dec1(z))
        logits = self.dec2(h2)
        return logits

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("sample_data","papers.json"))
    ap.add_argument("--vocab", type=int, default=5000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    # Load abstracts
    print(f"Loading abstracts from {os.path.basename(args.data)}...")
    abstracts = load_abstracts(args.data)
    print(f"Found {len(abstracts)} abstracts")

    # Build vocabulary
    print("Building vocabulary from words...")
    w2i, vocab, total_words = build_vocab(abstracts, args.vocab)
    V = len(vocab)
    print(f"Vocabulary size: {V} words")

    # Print architecture and parmaent count
    H, E = args.hidden, args.embed
    print(f"Model architecture: {V} -> {H} -> {E} -> {H} -> {V}")
    model = BoWAutoencoder(V, H, E)
    params = param_count(model)
    ok = "✓" if params < 2_000_000 else "✗"
    print(f"Total parameters: {params:,} (under 2,000,000 limit {ok})\n")

    # Vectorize data
    X = vectorize(abstracts, w2i, binary=True)
    ds = TensorDataset(X, X)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

    # Train
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    print("Training autoencoder...")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        avg = running / len(ds)
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs}, Loss: {avg:.4f}")
    dur = time.time() - start
    print(f"\nTraining complete in {dur:.1f} seconds")

if __name__ == "__main__":
    main()

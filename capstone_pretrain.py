"""
Capstone — Stage 2: pre-train the encoder.

BERT-style masked-token training on ALL events. Labels (fraud/normal) are
NEVER used here — this is fully self-supervised.

Saves the encoder weights (embedding + transformer) to encoder.pt.
"""
import random
import torch
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)

D_MODEL  = 32
N_HEADS  = 2
N_LAYERS = 2
LR       = 3e-3
STEPS    = 2000
BATCH    = 64
MASK_P   = 0.20


class Encoder(nn.Module):
    def __init__(self, V, d=D_MODEL, heads=N_HEADS, layers=N_LAYERS, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.enc(self.emb(x) + self.pos(positions))


class MLMHead(nn.Module):
    def __init__(self, V, d=D_MODEL):
        super().__init__()
        self.head = nn.Linear(d, V)

    def forward(self, h):
        return self.head(h)


def mask_batch(X_batch, mask_id, key_ids, p=MASK_P):
    X = X_batch.clone()
    y = torch.full_like(X, -100)
    is_value = ~torch.isin(X, key_ids)         # don't mask key tokens
    pick = (torch.rand_like(X, dtype=torch.float) < p) & is_value
    y[pick] = X[pick]
    X[pick] = mask_id
    return X, y


def main():
    blob = torch.load("data.pt")
    X, vocab, tok2id = blob["X"], blob["vocab"], blob["tok2id"]
    V = len(vocab)
    mask_id = tok2id["<mask>"]
    key_ids = torch.tensor([tok2id[k] for k in ["type", "amount", "time"]])

    encoder = Encoder(V)
    head = MLMHead(V)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    N = X.size(0)
    print(f"Pre-training on {N} users for {STEPS} steps...")
    for step in range(STEPS):
        idx = torch.randint(0, N, (BATCH,))
        x, y = mask_batch(X[idx], mask_id, key_ids)
        h = encoder(x)
        logits = head(h)
        loss = loss_fn(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == STEPS - 1:
            print(f"  step {step:4d}   loss {loss.item():.3f}")

    torch.save(encoder.state_dict(), "encoder.pt")
    print("Saved encoder weights -> encoder.pt")


if __name__ == "__main__":
    main()

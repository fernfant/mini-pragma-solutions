"""
Capstone — Stage 3 + 4: classifier head + baseline comparison.

For several training-set sizes, we train two classifiers:
  A. Frozen pre-trained encoder + linear head  (the foundation-model recipe)
  B. Random-init encoder + linear head trained end-to-end  (no pre-training)

We expect A to win when labels are scarce.
"""
import torch
import torch.nn as nn
from capstone_pretrain import Encoder, D_MODEL


SIZES = [20, 50, 200, 1000]
EPOCHS = 200
LR = 3e-3


class Classifier(nn.Module):
    def __init__(self, encoder, d=D_MODEL):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        h = self.encoder(x)            # (B, L, d)
        pooled = h.mean(dim=1)         # mean-pool over positions
        return self.head(pooled)


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def metrics(logits, y):
    pred = logits.argmax(-1)
    acc = (pred == y).float().mean().item()
    fraud_mask = (y == 1)
    if fraud_mask.sum() == 0:
        recall = float("nan")
    else:
        recall = (pred[fraud_mask] == 1).float().mean().item()
    return acc, recall


def train_classifier(clf, X_tr, y_tr, X_te, y_te, epochs=EPOCHS):
    params = [p for p in clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        logits = clf(X_tr)
        loss = loss_fn(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        clf.eval()
        return metrics(clf(X_te), y_te)


def main():
    torch.manual_seed(0)
    blob = torch.load("data.pt")
    X, y, vocab = blob["X"], blob["y"], blob["vocab"]
    V = len(vocab)

    n = X.size(0)
    perm = torch.randperm(n)
    split = int(n * 0.8)
    tr_idx, te_idx = perm[:split], perm[split:]
    X_te, y_te = X[te_idx], y[te_idx]

    print(f"Test set: {len(te_idx)} users ({int(y_te.sum())} fraud)\n")
    print(f"{'train':>6} | {'pretrained acc':>14} {'pretrained recall':>17} | {'baseline acc':>12} {'baseline recall':>15}")
    print("-" * 80)

    for size in SIZES:
        sub = tr_idx[:size]
        X_tr, y_tr = X[sub], y[sub]

        # A. frozen pre-trained encoder
        enc_a = Encoder(V)
        enc_a.load_state_dict(torch.load("encoder.pt"))
        freeze(enc_a)
        clf_a = Classifier(enc_a)
        acc_a, rec_a = train_classifier(clf_a, X_tr, y_tr, X_te, y_te)

        # B. random-init encoder, trained end-to-end
        torch.manual_seed(size)   # different seed per run for variety
        enc_b = Encoder(V)
        clf_b = Classifier(enc_b)
        acc_b, rec_b = train_classifier(clf_b, X_tr, y_tr, X_te, y_te)

        print(f"{size:>6} | {acc_a:>14.3f} {rec_a:>17.3f} | {acc_b:>12.3f} {rec_b:>15.3f}")


if __name__ == "__main__":
    main()

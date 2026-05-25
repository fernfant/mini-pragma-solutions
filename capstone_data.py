"""
Capstone — Stage 1: synthetic data.

Generates 2000 users (95% normal, 5% fraud), 20 events each.
Each event has three fields: type, amount, time.

Fraud signal is intentionally CONTEXTUAL — bursts of small night-time
test purchases with almost no logins — so the model needs to look at
combinations of fields and event order, not just token frequencies.
"""
import random
import torch

random.seed(42)

KEYS   = ["type", "amount", "time"]
TYPES  = ["purchase", "transfer", "login", "signup"]
AMTS   = ["tiny", "small", "medium", "large"]
TIMES  = ["morning", "afternoon", "evening", "night"]
VALUES = TYPES + AMTS + TIMES

PAD, MASK = "<pad>", "<mask>"
vocab = [PAD, MASK] + KEYS + VALUES
tok2id = {t: i for i, t in enumerate(vocab)}
V = len(vocab)

EVENTS_PER_USER = 20
N_USERS = 2000
FRAUD_RATE = 0.05


def normal_event():
    t = random.choices(TYPES,  weights=[6, 3, 2, 0.1])[0]
    a = random.choices(AMTS,   weights=[1, 5, 4, 2])[0]
    h = random.choices(TIMES,  weights=[3, 4, 4, 1])[0]
    return [("type", t), ("amount", a), ("time", h)]


def fraud_event(burst):
    if burst:
        t = "purchase"
        a = random.choices(["tiny", "small"], weights=[8, 2])[0]
        h = random.choices(["night", "evening"], weights=[8, 2])[0]
    else:
        t = random.choices(TYPES,  weights=[5, 4, 0.2, 0.5])[0]
        a = random.choices(AMTS,   weights=[2, 2, 2, 6])[0]
        h = random.choices(TIMES,  weights=[1, 1, 2, 6])[0]
    return [("type", t), ("amount", a), ("time", h)]


def normal_user():
    return [normal_event() for _ in range(EVENTS_PER_USER)]


def fraud_user():
    burst_start = random.randint(0, EVENTS_PER_USER - 8)
    return [fraud_event(burst_start <= i < burst_start + 6) for i in range(EVENTS_PER_USER)]


def encode_event(event):
    ids = []
    for k, v in event:
        ids.append(tok2id[k])
        ids.append(tok2id[v])
    return ids


def encode_user(events):
    return [tok for e in events for tok in encode_event(e)]


def main():
    users, labels = [], []
    for _ in range(N_USERS):
        if random.random() < FRAUD_RATE:
            users.append(fraud_user()); labels.append(1)
        else:
            users.append(normal_user()); labels.append(0)

    X = torch.tensor([encode_user(u) for u in users], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    torch.save({"X": X, "y": y, "vocab": vocab, "tok2id": tok2id}, "data.pt")

    print(f"Saved {len(users)} users to data.pt")
    print(f"  shape X: {tuple(X.shape)}   ({X.size(1)} tokens per user)")
    print(f"  labels:  {int(y.sum())} fraud, {int((y == 0).sum())} normal")

    def show(uid):
        evts = users[uid]
        print(f"\nUser {uid} ({'FRAUD' if labels[uid] else 'normal'}):")
        for e in evts[:8]:
            print("  ", " | ".join(f"{k}={v}" for k, v in e))
        print("   ...")

    show(next(i for i, l in enumerate(labels) if l == 0))
    show(next(i for i, l in enumerate(labels) if l == 1))


if __name__ == "__main__":
    main()

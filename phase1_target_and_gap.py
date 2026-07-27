"""
Phase 1 — Label-Only MI reproduction
------------------------------------
Goal: train a (deliberately overfit) target CNN on a subset of CIFAR-10, then run
the GAP ATTACK baseline — predict "member" iff the model classifies the point correctly.
This is the simplest label-only attack and the baseline every fancier attack must beat.

Run on Google Colab: Runtime > Change runtime type > GPU. torchvision auto-downloads CIFAR-10.
(Primer: ../../Study/label-only-mi-primer.md — see the "gap attack" + "why MI works" sections.)
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---- Data ---------------------------------------------------------------
# CIFAR-10 train = 50k images. We carve a disjoint MEMBER / NON-MEMBER split.
# The target model trains ONLY on members; non-members are never seen. The
# behavioural difference between them is exactly what membership inference exploits.
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
full = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)

N_MEMBERS, N_NONMEMBERS = 5000, 5000   # small members set -> model overfits -> bigger gap -> easier MI
perm = np.random.permutation(len(full))
member_idx    = perm[:N_MEMBERS]
nonmember_idx = perm[N_MEMBERS:N_MEMBERS + N_NONMEMBERS]

member_set    = Subset(full, member_idx)
nonmember_set = Subset(full, nonmember_idx)
train_loader  = DataLoader(member_set, batch_size=128, shuffle=True)

# ---- Model (a small CNN, ~ the paper's 4-layer conv net) ----------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1  = nn.Conv2d(3, 32, 3, padding=1)
        self.c2  = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SmallCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- Train (long, NO augmentation -> we WANT it to overfit) -------------
EPOCHS = 50
for ep in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        opt.step()
    if (ep + 1) % 10 == 0:
        print(f"epoch {ep + 1}/{EPOCHS}")

# ---- Label-only signal: was each point classified correctly? ------------
@torch.no_grad()
def correct_mask(dataset):
    model.eval()
    out = []
    for xb, yb in DataLoader(dataset, batch_size=256):
        xb, yb = xb.to(device), yb.to(device)
        out.append((model(xb).argmax(1) == yb).cpu())
    return torch.cat(out).numpy()

member_correct    = correct_mask(member_set)
nonmember_correct = correct_mask(nonmember_set)
train_acc, test_acc = member_correct.mean(), nonmember_correct.mean()
print(f"train(member) acc {train_acc:.3f} | test(non-member) acc {test_acc:.3f} | GAP {train_acc - test_acc:.3f}")

# ---- Gap Attack ---------------------------------------------------------
# Predict member(1) iff correctly classified. Evaluate on a BALANCED set.
y_true = np.concatenate([np.ones(N_MEMBERS), np.zeros(N_NONMEMBERS)])
y_pred = np.concatenate([member_correct, nonmember_correct]).astype(int)
mi_acc = (y_pred == y_true).mean()
theory = 0.5 + (train_acc - test_acc) / 2
print(f"GAP-ATTACK MI accuracy: {mi_acc:.3f}  (theory 1/2+(acc_tr-acc_te)/2 = {theory:.3f})")

# ---- Save (target model + splits + results) for later phases ------------
# Phase 2 (augmentation) reuses the SAME target model + member/non-member indices,
# and we keep raw results so we can re-analyse offline (same habit as the evals repo).
torch.save(model.state_dict(), "target_model.pt")
np.savez("phase1_results.npz",
         member_idx=member_idx, nonmember_idx=nonmember_idx,
         member_correct=member_correct, nonmember_correct=nonmember_correct,
         train_acc=train_acc, test_acc=test_acc, mi_acc=mi_acc)
print("saved target_model.pt + phase1_results.npz")

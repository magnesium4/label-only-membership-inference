"""
Phase 2 — Label-Only MI reproduction
------------------------------------
The paper's actual contribution: recover a confidence-like signal using ONLY hard
labels, by asking how robust a point's predicted label is to small augmentations.

Idea: for each point, query the target on N augmented copies (rotations + shifts)
and record a binary vector of correct/incorrect. Members should stay correct under
perturbation more often than non-members. Train a shallow classifier on those
vectors -> predict membership. It has to beat the Phase 1 gap baseline (0.719).

Reuses Phase 1's saved target_model.pt + the SAME member/non-member indices, so the
comparison is apples-to-apples.

  !! EXPECTATION-SETTING !!
  Our Phase 1 target was trained WITHOUT augmentation (on purpose, to overfit).
  The augmentation attack is strongest when the target was trained WITH the same
  augmentations, because then members are specifically robust to them. So this may
  only modestly beat the gap baseline. That result is itself the answer to the
  question of what happens with models not trained with augmentation.

Run on Google Colab: Runtime > Change runtime type > GPU.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from sklearn.linear_model import LogisticRegression

torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---- Data + the SAME splits as Phase 1 ----------------------------------
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
full = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)

ph1 = np.load("phase1_results.npz")
member_idx, nonmember_idx = ph1["member_idx"], ph1["nonmember_idx"]
member_set    = Subset(full, member_idx)
nonmember_set = Subset(full, nonmember_idx)
print(f"members {len(member_idx)} | non-members {len(nonmember_idx)}")
print(f"Phase 1 baseline: gap-attack MI acc {ph1['mi_acc']:.3f}")

# ---- Target model (identical architecture, frozen) ----------------------
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
model.load_state_dict(torch.load("target_model.pt", map_location=device))
model.eval()

# ---- The augmentation set ------------------------------------------------
# Paper-style: 2r+1 rotations and 4d+1 translations. Index 0 is the IDENTITY,
# so column 0 of the feature matrix is exactly the Phase 1 gap-attack signal.
ROT_R, TRANS_D = 15, 4          # degrees, pixels (Phase 4 will tune these on shadow models)

augs = [("identity", None)]
augs += [("rotate", float(a)) for a in range(-ROT_R, ROT_R + 1) if a != 0]
for d in range(1, TRANS_D + 1):
    augs += [("translate", (d, 0)), ("translate", (-d, 0)),
             ("translate", (0, d)), ("translate", (0, -d))]
print(f"{len(augs)} queries per point "
      f"({2*ROT_R+1} rotations + {4*TRANS_D} translations)")

def apply_aug(xb, kind, param):
    if kind == "identity":
        return xb
    if kind == "rotate":
        return TF.rotate(xb, param)
    dx, dy = param
    return TF.affine(xb, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])

@torch.no_grad()
def correctness_matrix(dataset):
    """-> bool array (n_points, n_augs): was each augmented copy classified correctly?"""
    cols = [[] for _ in augs]
    for xb, yb in DataLoader(dataset, batch_size=256):
        xb, yb = xb.to(device), yb.to(device)
        for j, (kind, param) in enumerate(augs):
            pred = model(apply_aug(xb, kind, param)).argmax(1)
            cols[j].append((pred == yb).cpu())
    return torch.stack([torch.cat(c) for c in cols], dim=1).numpy()

print("querying target on augmented copies...")
X_mem = correctness_matrix(member_set)
X_non = correctness_matrix(nonmember_set)
print("feature matrices:", X_mem.shape, X_non.shape)

# Sanity check: column 0 must reproduce Phase 1's per-point correctness exactly.
assert (X_mem[:, 0] == ph1["member_correct"]).all(), "identity column != Phase 1 members"
assert (X_non[:, 0] == ph1["nonmember_correct"]).all(), "identity column != Phase 1 non-members"
print("identity column matches Phase 1 ✓")

print(f"mean augmented accuracy — members {X_mem.mean():.3f} | non-members {X_non.mean():.3f}")

# ---- Attack classifier ---------------------------------------------------
# The attack needs labelled membership examples to learn from. Here we grant the
# attacker half the points with known labels and evaluate on the held-out half.
# (Phase 4 removes this crutch: shadow models supply the labels instead.)
X = np.concatenate([X_mem, X_non]).astype(np.float32)
y = np.concatenate([np.ones(len(X_mem)), np.zeros(len(X_non))])

rng = np.random.default_rng(0)
def halves(n):
    p = rng.permutation(n); return p[: n // 2], p[n // 2 :]

m_tr, m_te = halves(len(X_mem))
n_tr, n_te = halves(len(X_non))
tr = np.concatenate([m_tr, len(X_mem) + n_tr])
te = np.concatenate([m_te, len(X_mem) + n_te])

clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
aug_acc = clf.score(X[te], y[te])

# ---- Compare fairly, on the SAME held-out points -------------------------
# Gap attack = "member iff the un-augmented point is correct" = column 0.
gap_acc_te = ((X[te][:, 0] == 1).astype(float) == y[te]).mean()
# A simple interpretable alternative: threshold on how many augmentations survived.
n_correct_tr, n_correct_te = X[tr].sum(1), X[te].sum(1)
best_t = max(range(len(augs) + 1),
             key=lambda t: (((n_correct_tr >= t).astype(float)) == y[tr]).mean())
count_acc = ((n_correct_te >= best_t).astype(float) == y[te]).mean()

print("\n---- Phase 2 results (held-out half) ----")
print(f"gap attack (baseline)          : {gap_acc_te:.3f}")
print(f"count-threshold  (>= {best_t:2d} correct): {count_acc:.3f}")
print(f"augmentation attack (logistic) : {aug_acc:.3f}")
print(f"improvement over gap           : {aug_acc - gap_acc_te:+.3f}")
print("BEATS the gap baseline ✓" if aug_acc > gap_acc_te else "does NOT beat the gap baseline ✗")

# ---- Save raw results; analyse offline later (same habit as the evals repo) ----
np.savez("phase2_results.npz",
         X_mem=X_mem, X_non=X_non,
         member_idx=member_idx, nonmember_idx=nonmember_idx,
         aug_kinds=np.array([k for k, _ in augs]),
         aug_params=np.array([str(p) for _, p in augs]),
         train_idx=tr, test_idx=te,
         rot_r=ROT_R, trans_d=TRANS_D,
         gap_acc_te=gap_acc_te, count_acc=count_acc, aug_acc=aug_acc, best_t=best_t)
print("saved phase2_results.npz")

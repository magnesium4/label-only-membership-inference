# Label-Only Membership Inference — a reproduction

Reproducing **Choquette-Choo, Tramèr, Carlini & Papernot, _"Label-Only Membership Inference Attacks"_** (ICML 2021, [arXiv:2007.14321](https://arxiv.org/abs/2007.14321)).

**The claim under test:** even with access to nothing but hard predicted labels — no confidence scores — an attacker can infer training-set membership about as well as prior confidence-based attacks, by measuring how robust a point's classification is to small perturbations. Members are more robust because the model overfit to them.

Write-ups of each phase are at [mgohar.com](https://mgohar.com) (series: *Label-Only Membership Inference*).

## Setup

CIFAR-10, with a disjoint **5,000 member / 5,000 non-member** split. The target is a small CNN (two conv layers, two fully-connected), trained for 50 epochs on the members only and deliberately allowed to overfit — the train/test gap is what membership inference exploits. Trained on a Colab GPU; the attacks themselves are CPU-friendly.

## Results

### Phase 1 — gap attack (baseline)

Predict "member" iff the model classifies the point correctly.

```
train acc 1.000 | test acc 0.562 | gap 0.438
gap-attack MI accuracy: 0.719
```

This matches the theoretical `0.5 + gap/2 = 0.719` exactly.

### Phase 2 — data-augmentation attack

For each point, query the target on 47 augmented copies (the original, 30 rotations from −15° to +15°, and 16 translations of 1–4 px along each axis) and record whether each copy is still classified **correctly**. Fit a logistic regression on the resulting 47-bit vectors. The attacker is given true membership labels for half the points and scored on the held-out half.

```
mean augmented accuracy — members 0.770 | non-members 0.513

gap attack (baseline)          : 0.720
count-threshold (>= 20 correct): 0.685
augmentation attack (logistic) : 0.759
improvement over gap           : +0.039
```

The augmentation attack beats the gap baseline, reproducing the paper's core claim.

**The interesting wrinkle:** the naive **count-threshold** variant — call a point a member if enough of its 47 copies survive — scores **0.685, worse than the gap attack**, even though the threshold is swept and the best one on the attacker's training half is used. The augmented gap (0.770 − 0.513 = 0.257) is much smaller than the raw gap (0.438), so augmentation-robustness on its own is a weaker signal than plain correctness against a target trained without augmentation. The logistic model wins by *weighting* the queries rather than averaging them.

## Notes on deviations from the paper

Recorded so the comparison is honest:

1. **Different augmentation family.** The paper generates `N=3` rotations (`{−r, 0, +r}` for `r ∈ [1,15]`) and `N=4d+1` translations satisfying `|i|+|j|=d` — one fixed shift distance, every direction at it, diagonals included. This implementation steps rotations every degree from −15 to +15 and sweeps translation distances `d=1..4` along the axes only, so it never generates a diagonal shift. The two schemes coincide at `d=1` and diverge from `d=2`.
2. **Some queries fall outside the paper's working range.** §5.5 reports the attack only clears the baseline for `1 ≤ r ≤ 8` and `1 ≤ d ≤ 2`, since small perturbations rarely change predictions and large ones cause misclassifications regardless of membership. Using `r=15, d=4` puts 22 of the 47 queries outside that window — a competing explanation for the modest gain that is not yet ruled out.
3. **Effect size is in line with the paper.** §5.5 reports 3–4 percentage points for an optimal `r`/`d`; this run gives +3.9. The paper's Figure 2 targets are trained on 2,500 points rather than 5,000, so the overfitting regimes differ and the comparison is suggestive rather than exact.
4. **The attacker is handed ground-truth membership labels** for half the points, which the real threat model doesn't allow. Shadow models remove that crutch in a later phase.

## Files

| file | what it does |
|---|---|
| `phase1_target_and_gap.py` | trains the target CNN, saves it, runs the gap-attack baseline |
| `phase2_augmentation_attack.py` | builds the 47-query correctness matrix, fits the attack classifier, saves raw vectors to `phase2_results.npz` |

Raw per-point results are saved so the analysis can be re-run offline without touching a GPU.

## Status

- [x] Phase 1 — target model + gap baseline
- [x] Phase 2 — data-augmentation attack
- [ ] Offline analysis of Phase 2 — logistic coefficients (does the identity query dominate?), a paired McNemar test on the +0.039, accuracy vs. number of queries, and a refit on the in-range queries only
- [ ] Phase 3 — boundary-distance attack
- [ ] Phase 4 — shadow models, to remove the ground-truth-label assumption
- [ ] Phase 5 — evaluation and write-up

**Planned extension:** re-score the attacks under the **TPR at low FPR** lens argued for in Carlini et al. 2022, [*Membership Inference Attacks From First Principles*](https://arxiv.org/abs/2112.03570). This paper reports balanced accuracy throughout, and an attack averaging 0.759 can still be useless at the 0.1% false-positive rate a real attacker would operate at. That number does not appear to have been reported for these attacks.

## Running it

```bash
pip install torch torchvision numpy scikit-learn
python phase1_target_and_gap.py    # trains the target, writes the checkpoint
python phase2_augmentation_attack.py
```

`torchvision` downloads CIFAR-10 on first run. Phase 1 wants a GPU; Phase 2 is fast either way.

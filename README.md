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
train acc 1.000 | test acc 0.564 | gap 0.436
gap-attack MI accuracy: 0.718
```

This matches the theoretical `0.5 + gap/2 = 0.718` exactly.

### Phase 2 — data-augmentation attack

For each point, query the target on 47 augmented copies (the original, 30 rotations from −15° to +15°, and 16 translations of 1–4 px along each axis) and record whether each copy is still classified **correctly**. Fit a logistic regression on the resulting 47-bit vectors. The attacker is given true membership labels for half the points and scored on the held-out half.

```
mean augmented accuracy — members 0.771 | non-members 0.516

gap attack (baseline)          : 0.719
count-threshold (>= 23 correct): 0.683
augmentation attack (logistic) : 0.750
improvement over gap           : +0.031
```

The augmentation attack beats the gap baseline, reproducing the paper's core claim.

**The interesting wrinkle:** the naive **count-threshold** variant — call a point a member if enough of its 47 copies survive — scores **0.683, worse than the gap attack**, even though the threshold is swept and the best one on the attacker's training half is used. The augmented gap (0.771 − 0.516 = 0.255) is much smaller than the raw gap (0.436), so augmentation-robustness on its own is a weaker signal than plain correctness against a target trained without augmentation. The logistic model wins by *weighting* the queries rather than averaging them.

### Offline analysis of Phase 2

Ablations and a paired significance test, run from the saved vectors without retraining anything. The refit reproduces the published 0.7498 exactly, so these are read off the same model that produced it.

```
subset                                     n     acc   vs gap  of gain
all queries (published attack)            47  0.7498  +0.0310    100%
identity only (= gap attack)               1  0.7188  +0.0000      0%
drop identity group                       44  0.7490  +0.0302     97%
near cluster, as 11 columns               11  0.7436  +0.0248     80%
near cluster, duplicates removed           9  0.7422  +0.0234     75%
near cluster, no identity at all           8  0.7422  +0.0234     75%
rotations |r| in 2..3 only                 4  0.7416  +0.0228     74%
translations d=1 only                      4  0.7160  -0.0028     -9%
drop near cluster                         36  0.7178  -0.0010     -3%
paper in-range only (1<=r<=8, 1<=d<=2)    25  0.7478  +0.0290     94%
out-of-range only                         23  0.7188  +0.0000      0%
```

**The gain is concentrated in a handful of small rotations, and the un-augmented query is redundant.** Four rotations — `±2°` and `±3°` — recover 74% of the effect on their own, without the un-augmented query at all. Widening to the nine distinct queries within `|r| ≤ 3, d ≤ 1` reaches 75%, and the remaining 36 land marginally below the one-query gap baseline. Removing identity costs 0.0008 against the full set and *nothing at all* inside the near cluster, because `rotate ±2` agree with it on 98% of points and stand in for it. The `d=1` translations contribute 0.0006 on top of the rotations and score below the gap baseline alone, so on this target the signal is rotational. Large rotations carry *negative* weight — surviving a 14° rotation is evidence against membership. That is why equal-weight counting loses: it lets three dozen uninformative and sign-flipped columns outvote the four that carry signal.

The near cluster is listed three ways deliberately. Counting it as eleven columns double-counts the two `rotate ±1` duplicates, and it also scores higher (0.7436 against 0.7422) purely because three identical columns divide the L2 penalty between them and so face a weaker one. The deduplicated row is the honest number.

**The margin is significant.** McNemar on the same held-out points: 4,707 of 5,000 are scored identically by both attacks, and of the 293 disagreements the augmentation attack wins 224 to 69. Exact two-sided `p = 2.8e-20`, winning 71–81% of disagreements at 95% confidence. The unpaired two-proportion test on the same numbers gives only `z = 3.51`, so discarding the pairing costs sixteen orders of magnitude. This conditions on the trained target and the saved split, so it addresses sampling noise over test points and not the retraining variance in deviation 4 below.

**Two queries are wasted.** `rotate ±1°` returns the image unchanged: nearest-neighbour interpolation on a 32×32 grid displaces pixels by at most 0.383 px at 1°, below the half-pixel rounding threshold. Those columns are bit-identical to identity on all 10,000 points, so the effective budget is **45 queries, not 47**. It also distorts the raw coefficients, since the L2 penalty splits the gap signal's weight across all three copies — deduplicated, identity fits at +2.14 rather than +1.01.

## Notes on deviations from the paper

Recorded so the comparison is honest:

1. **Different augmentation family.** The paper generates `N=3` rotations (`{−r, 0, +r}` for `r ∈ [1,15]`) and `N=4d+1` translations satisfying `|i|+|j|=d` — one fixed shift distance, every direction at it, diagonals included. This implementation steps rotations every degree from −15 to +15 and sweeps translation distances `d=1..4` along the axes only, so it never generates a diagonal shift. The two schemes coincide at `d=1` and diverge from `d=2`.
2. **Some queries fall outside the paper's working range.** §5.5 reports the attack only clears the baseline for `1 ≤ r ≤ 8` and `1 ≤ d ≤ 2`, since small perturbations rarely change predictions and large ones cause misclassifications regardless of membership. Using `r=15, d=4` puts 22 of the 47 queries outside that window. This was a competing explanation for the modest gain; the in-range ablation above rules it out, since restricting to the paper's window scores 0.7478 against 0.7498 for the full set.
3. **Effect size is in line with the paper.** §5.5 reports 3–4 percentage points for an optimal `r`/`d`; this run gives +3.1, at the bottom of that range. The paper's Figure 2 targets are trained on 2,500 points rather than 5,000, so the overfitting regimes differ and the comparison is suggestive rather than exact.
4. **The exact digits are not stable across runs.** Both scripts set `torch.manual_seed(0)` and `np.random.seed(0)`, but GPU training is nondeterministic, so retraining the target shifts everything slightly. A rerun moved the headline gain from +0.039 to +0.031 — about a fifth of the effect — while every qualitative conclusion held. Treat the third decimal as noise, and read the paired significance test rather than the point estimate.
5. **The attacker is handed ground-truth membership labels** for half the points, which the real threat model doesn't allow. Shadow models remove that crutch in a later phase.

## Files

| file | what it does |
|---|---|
| `phase1_target_and_gap.py` | trains the target CNN, saves it, runs the gap-attack baseline |
| `phase2_augmentation_attack.py` | builds the 47-query correctness matrix, fits the attack classifier, saves raw vectors to `phase2_results.npz` |
| `phase2_analysis.py` | offline: refits from the saved vectors, reports coefficients, duplicate queries, ablations and the McNemar test |

Raw per-point results are saved so the analysis can be re-run offline without touching a GPU.

## Status

- [x] Phase 1 — target model + gap baseline
- [x] Phase 2 — data-augmentation attack
- [x] Offline analysis of Phase 2 — coefficients, ablations, in-range refit, paired McNemar
- [ ] Accuracy vs. number of queries — how much of the gain survives on a smaller budget, given that 11 queries already recover 80% of it
- [ ] Phase 3 — boundary-distance attack
- [ ] Phase 4 — shadow models, to remove the ground-truth-label assumption
- [ ] Phase 5 — evaluation and write-up

**Planned extension:** re-score the attacks under the **TPR at low FPR** lens argued for in Carlini et al. 2022, [*Membership Inference Attacks From First Principles*](https://arxiv.org/abs/2112.03570). This paper reports balanced accuracy throughout, and an attack averaging 0.750 can still be useless at the 0.1% false-positive rate a real attacker would operate at. That number does not appear to have been reported for these attacks.

## Running it

```bash
pip install torch torchvision numpy scikit-learn
python phase1_target_and_gap.py    # trains the target, writes the checkpoint
python phase2_augmentation_attack.py
python phase2_analysis.py          # offline, reads phase2_results.npz
```

`torchvision` downloads CIFAR-10 on first run. Phase 1 wants a GPU; Phase 2 is fast either way. `phase2_analysis.py` needs only `numpy`, `scipy` and `scikit-learn`, since it never touches the target model.

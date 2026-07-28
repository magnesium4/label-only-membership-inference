"""
Phase 2 analysis — Label-Only MI reproduction
---------------------------------------------
Offline post-processing of `phase2_results.npz`. Every number here is derived from the correctness matrices that the
Phase 2 run already saved.

The attack classifier itself was never serialised, only its score was. So the
first job is to rebuild the fit from the saved split and confirm it reproduces
the published accuracy exactly. Coefficients, ablations and significance tests
are all read off that refit, so they are worth precisely as much as the match is.

Run:  python3 phase2_analysis.py
"""
import ast

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression

RESULTS_PATH = "phase2_results.npz"
MAX_ITER = 1000          # matches the Phase 2 fit; changing it changes the model
REFIT_TOLERANCE = 1e-12  # lbfgs is deterministic, so only float noise is allowed
IDENTITY_COL = 0         # column 0 is the un-augmented query = the gap signal
TOP_N = 10               # how many queries to show at each end of the ranking


def load_results(path=RESULTS_PATH):
    """Open the saved Phase 2 archive.

    `np.load` on an .npz is lazy — it returns a dict-like handle and reads each
    array only when indexed.
    """
    return np.load(path)


def describe_arrays(results):
    """Print the name, shape and dtype of every array in the archive.

    Scalars come back as 0-dimensional arrays (shape `()`), not Python floats,
    which is why they need an explicit `float()`/`int()` before comparison.
    """
    print("---- contents of", RESULTS_PATH, "----")
    for name in results.files:
        array = results[name]
        print(f"  {name:<14} {str(array.shape):<12} {array.dtype}")


def print_saved_scores(results):
    """Print the headline accuracies recorded by the Phase 2 run."""
    print("\n---- saved Phase 2 scores (held-out half) ----")
    print(f"  gap attack (baseline)          : {float(results['gap_acc_te']):.3f}")
    print(f"  count-threshold (>= {int(results['best_t']):2d} correct): {float(results['count_acc']):.3f}")
    print(f"  augmentation attack (logistic) : {float(results['aug_acc']):.3f}")


def build_design_matrix(results):
    """Stack the member / non-member correctness matrices into features + labels.

    Rows are members first, then non-members — the layout the saved split
    indexes into. The float32 cast mirrors Phase 2; matching it keeps the refit
    bit-identical.

    Returns (features, labels) of shape (n_points, n_queries) and (n_points,),
    where labels are 1 for members and 0 for non-members.
    """
    member_correct = results["X_mem"]
    nonmember_correct = results["X_non"]
    features = np.concatenate([member_correct, nonmember_correct]).astype(np.float32)
    labels = np.concatenate([np.ones(len(member_correct)),
                             np.zeros(len(nonmember_correct))])
    return features, labels


def load_split(results, n_points):
    """Return the saved train/test row indices, checking they partition the rows.

    The split is deliberately *not* re-derived from the RNG: Phase 2 drew the
    member and non-member halves from one shared generator, so reproducing it
    means replaying both draws in order. The saved indices are the ground truth,
    and the non-member half already carries its offset into the stacked matrix.
    """
    train_idx, test_idx = results["train_idx"], results["test_idx"]
    assert len(train_idx) == len(test_idx) == n_points // 2, "halves are unequal"
    assert np.intersect1d(train_idx, test_idx).size == 0, "train/test overlap"
    assert np.union1d(train_idx, test_idx).size == n_points, "split misses rows"
    return train_idx, test_idx


def refit_attack_classifier(features, labels, train_idx):
    """Refit Phase 2's logistic attack on the training half.

    The default lbfgs solver is deterministic and never touches a random seed,
    so — unlike the GPU training upstream — this has no licence to drift between
    runs.
    """
    return LogisticRegression(max_iter=MAX_ITER).fit(features[train_idx],
                                                     labels[train_idx])


def check_reproduction(classifier, features, labels, test_idx, expected_acc):
    """Verify the refit scores what Phase 2 published, and return its accuracy.

    A mismatch means the reconstruction diverged from the original run, and
    every coefficient read off this model would describe a classifier that never
    produced the published result.
    """
    accuracy = classifier.score(features[test_idx], labels[test_idx])
    delta = accuracy - expected_acc
    print("\n---- refit reproduction check ----")
    print(f"  refit accuracy : {accuracy:.6f}")
    print(f"  saved accuracy : {expected_acc:.6f}")
    print(f"  difference     : {delta:+.2e}")
    assert abs(delta) < REFIT_TOLERANCE, "refit does not reproduce the saved score"
    print("  reproduces the saved augmentation attack ✓")
    return accuracy


def query_names(results):
    """Readable name for each of the 47 queries, in column order.

    Column 0 is the identity (un-augmented) query; its saved parameter is the
    string "None", so it is named on its own.
    """
    return [kind if kind == "identity" else f"{kind} {param}"
            for kind, param in zip(results["aug_kinds"], results["aug_params"])]


def rank_of(values, column):
    """1-based rank of `column` when `values` is sorted descending by magnitude."""
    order = np.argsort(-np.abs(values))
    return int(np.flatnonzero(order == column)[0]) + 1


def summarise_coefficients(classifier, features, train_idx, names):
    """Report which queries the logistic attack actually leans on.

    Two views, because neither alone is honest:

    * the raw coefficient — comparable across columns here only because every
      feature is binary on the same 0/1 scale, so no standardisation is needed;
    * the coefficient scaled by the feature's training spread — a query that
      almost every point survives carries little information however large its
      coefficient, and this view prices that in.

    Where the two rankings disagree, the disagreement is the finding.

    Caveat worth carrying into the write-up: the 47 columns are heavily
    correlated (a point correct under identity is usually correct under a 2°
    rotation), and correlated features make individual logistic coefficients
    unstable. This says what the fit looks like, not what it depends on — the
    ablation is what settles dependence.
    """
    weights = classifier.coef_[0]
    spread = features[train_idx].std(axis=0)
    contributions = weights * spread

    print("\n---- logistic coefficients ----")
    print(f"  intercept: {classifier.intercept_[0]:+.4f}")
    print(f"  identity coefficient  : {weights[IDENTITY_COL]:+.4f}"
          f"  (rank {rank_of(weights, IDENTITY_COL)} of {len(weights)} by |coef|)")
    print(f"  identity contribution : {contributions[IDENTITY_COL]:+.4f}"
          f"  (rank {rank_of(contributions, IDENTITY_COL)} of {len(contributions)} by |coef x spread|)")
    print(f"  identity share of total |coef|: "
          f"{abs(weights[IDENTITY_COL]) / np.abs(weights).sum():.1%}")

    order = np.argsort(-weights)
    print(f"\n  top {TOP_N} by signed coefficient (evidence FOR membership):")
    for col in order[:TOP_N]:
        print(f"    {names[col]:<16} {weights[col]:+.4f}   x spread {contributions[col]:+.4f}")
    print(f"\n  bottom {TOP_N} by signed coefficient (evidence AGAINST membership):")
    for col in order[-TOP_N:]:
        print(f"    {names[col]:<16} {weights[col]:+.4f}   x spread {contributions[col]:+.4f}")

    print("\n  by augmentation family:")
    kinds = np.array([name.split()[0] for name in names])
    for kind in ["identity", "rotate", "translate"]:
        mask = kinds == kind
        print(f"    {kind:<10} n={mask.sum():<3} "
              f"sum|coef| {np.abs(weights[mask]).sum():6.3f}   "
              f"mean|coef| {np.abs(weights[mask]).mean():6.4f}")
    return weights


def find_duplicate_queries(features, names):
    """Find queries whose correctness column is bit-identical to another's.

    These are wasted queries: they cost the attacker a query to the target and
    return information already held. They also distort the coefficients, because
    the L2 penalty splits weight evenly across identical columns — so a signal
    with k copies looks 1/k as important as it is.

    Returns (keep_cols, groups): the first occurrence of each distinct column,
    and the duplicate groups as lists of names.
    """
    _, first_occurrence, membership = np.unique(features.T, axis=0,
                                                return_index=True,
                                                return_inverse=True)
    keep_cols = np.sort(first_occurrence)
    groups = [[names[col] for col in np.flatnonzero(membership == g)]
              for g in np.unique(membership)
              if (membership == g).sum() > 1]
    return keep_cols, groups


def refit_without_duplicates(features, labels, train_idx, test_idx, names):
    """Refit on distinct columns only, so coefficients are read off a full-rank fit.

    Accuracy should barely move — duplicate columns carry no extra information —
    but the coefficients become interpretable, since weight is no longer being
    divided among copies of the same query.
    """
    keep_cols, groups = find_duplicate_queries(features, names)
    print("\n---- duplicate queries ----")
    if not groups:
        print("  none; all queries are distinct")
        return None
    for group in groups:
        print(f"  identical columns: {', '.join(group)}")
    print(f"  {len(keep_cols)} distinct of {features.shape[1]} queries "
          f"({features.shape[1] - len(keep_cols)} wasted)")

    classifier = LogisticRegression(max_iter=MAX_ITER).fit(
        features[train_idx][:, keep_cols], labels[train_idx])
    accuracy = classifier.score(features[test_idx][:, keep_cols], labels[test_idx])
    print(f"  deduplicated refit accuracy: {accuracy:.4f}")

    weights = classifier.coef_[0]
    identity = int(np.flatnonzero(keep_cols == IDENTITY_COL)[0])
    print(f"  identity coefficient: {weights[identity]:+.4f} "
          f"(rank {rank_of(weights, identity)} of {len(weights)}, "
          f"{abs(weights[identity]) / np.abs(weights).sum():.1%} of total |coef|)")
    kept_names = [names[col] for col in keep_cols]
    print("  top 5 by signed coefficient:")
    for col in np.argsort(-weights)[:5]:
        print(f"    {kept_names[col]:<18} {weights[col]:+.4f}")
    return classifier


def query_magnitudes(results):
    """Perturbation size per query: rotation in degrees, translation in pixels.

    Identity is 0. Translations are stored as the string form of a (dx, dy)
    tuple, only ever axis-aligned here, so the magnitude is the non-zero entry.
    """
    magnitudes = []
    for kind, param in zip(results["aug_kinds"], results["aug_params"]):
        if kind == "identity":
            magnitudes.append(0.0)
        elif kind == "rotate":
            magnitudes.append(abs(float(param)))
        else:
            dx, dy = ast.literal_eval(param)
            magnitudes.append(float(max(abs(dx), abs(dy))))
    return np.array(magnitudes)


def score_subset(features, labels, train_idx, test_idx, columns):
    """Refit the attack on a subset of queries and score it on the held-out half.

    Same solver and penalty as the full attack, so the only thing varying across
    ablations is which queries the attacker is allowed to ask.
    """
    columns = np.atleast_1d(columns)
    classifier = LogisticRegression(max_iter=MAX_ITER).fit(
        features[train_idx][:, columns], labels[train_idx])
    return classifier.score(features[test_idx][:, columns], labels[test_idx])


def run_ablations(results, features, labels, train_idx, test_idx, names):
    """Measure what the attack actually depends on by removing queries.

    Coefficients describe a fit; ablations test dependence. With 47 heavily
    correlated columns only the second is trustworthy.

    Note the identity ablation drops the whole *group* of columns identical to
    identity, not just column 0. Dropping column 0 alone would leave rotate ±1,
    which are bit-identical to it, and the ablation would report no effect while
    the signal was still fully present.

    The near cluster is reported three ways for the same reason. Counting it as
    eleven columns double-counts the two duplicates, and it also scores higher
    (0.7436 against 0.7422) purely because three identical columns share the L2
    penalty between them and so face a weaker one. The deduplicated row is the
    honest number; the eleven-column row is kept to show the size of the
    artefact.
    """
    kinds = np.array([name.split()[0] for name in names])
    magnitude = query_magnitudes(results)
    identity_group = np.flatnonzero(
        (features == features[:, [IDENTITY_COL]]).all(axis=0))
    near = (magnitude <= 3) & (kinds == "rotate") | (magnitude <= 1) & (kinds == "translate")
    near[IDENTITY_COL] = True
    # The paper's working window: 1 <= r <= 8 rotations, 1 <= d <= 2 translations.
    # Identity is deliberately left OUT of both range masks so they partition the
    # 46 augmented columns exactly (24 in, 22 out). Each is then reported twice,
    # with and without identity added back: with it, the subset can fall back on
    # the gap signal, which is the fair comparison against the full attack; without
    # it, you see what those queries know on their own.
    in_range = (((kinds == "rotate") & (magnitude >= 1) & (magnitude <= 8))
                | ((kinds == "translate") & (magnitude >= 1) & (magnitude <= 2)))
    out_of_range = (((kinds == "rotate") & (magnitude >= 9))
                    | ((kinds == "translate") & (magnitude >= 3)))
    assert 1 + in_range.sum() + out_of_range.sum() == features.shape[1], \
        "the range masks plus identity must partition every query"

    all_cols = np.arange(features.shape[1])
    near_cols = np.flatnonzero(near)
    duplicates = identity_group[1:]          # the rotate +/-1 copies of identity
    small_rotations = np.flatnonzero((kinds == "rotate")
                                     & (magnitude >= 2) & (magnitude <= 3))
    one_pixel_shifts = np.flatnonzero((kinds == "translate") & (magnitude == 1))

    subsets = [
        ("all queries (published attack)", all_cols),
        ("identity only (= gap attack)", np.array([IDENTITY_COL])),
        ("drop identity group", np.setdiff1d(all_cols, identity_group)),
        ("near cluster, as 11 columns", near_cols),
        ("near cluster, duplicates removed", np.setdiff1d(near_cols, duplicates)),
        ("near cluster, no identity at all", np.setdiff1d(near_cols, identity_group)),
        ("rotations |r| in 2..3 only", small_rotations),
        ("translations d=1 only", one_pixel_shifts),
        ("drop near cluster", np.flatnonzero(~near)),
        ("paper in-range, with identity", np.union1d(np.flatnonzero(in_range),
                                                     [IDENTITY_COL])),
        ("paper in-range, no identity", np.setdiff1d(np.flatnonzero(in_range),
                                                     identity_group)),
        ("out-of-range, with identity", np.union1d(np.flatnonzero(out_of_range),
                                                   [IDENTITY_COL])),
        ("out-of-range, no identity", np.flatnonzero(out_of_range)),
    ]

    gap_baseline = float(results["gap_acc_te"])
    full_gain = score_subset(features, labels, train_idx, test_idx,
                             all_cols) - gap_baseline
    print("\n---- ablations (held-out half) ----")
    print(f"  {'subset':<40} {'n':>3}  {'acc':>6}  {'vs gap':>7}  {'of gain':>7}")
    for name, columns in subsets:
        accuracy = score_subset(features, labels, train_idx, test_idx, columns)
        gain = accuracy - gap_baseline
        print(f"  {name:<40} {len(columns):>3}  {accuracy:>6.4f}  "
              f"{gain:>+7.4f}  {gain / full_gain:>6.0%}")
    print(f"  (gap baseline = {gap_baseline:.4f}, full gain = {full_gain:+.4f})")


def mcnemar_test(classifier, features, labels, test_idx):
    """Paired significance test: does the augmentation attack beat the gap attack?

    Both attacks score the *same* held-out points, so the comparison is paired
    and an unpaired test would throw away that structure and overstate the
    uncertainty. McNemar looks only at the points where the two disagree: under
    the null that neither is better, each disagreement is a coin flip, so the
    discordant counts are Binomial(b + c, 0.5).

    The exact binomial form is used rather than the chi-square approximation —
    with a few hundred discordant pairs it costs nothing and needs no continuity
    correction.
    """
    y_true = labels[test_idx]
    gap_correct = (features[test_idx][:, IDENTITY_COL] == 1) == y_true
    aug_correct = classifier.predict(features[test_idx]) == y_true

    both = int((gap_correct & aug_correct).sum())
    gap_only = int((gap_correct & ~aug_correct).sum())
    aug_only = int((~gap_correct & aug_correct).sum())
    neither = int((~gap_correct & ~aug_correct).sum())
    discordant = gap_only + aug_only

    test = binomtest(aug_only, discordant, 0.5)
    print("\n---- McNemar: augmentation attack vs gap attack ----")
    print(f"  both correct           : {both}")
    print(f"  gap only               : {gap_only}")
    print(f"  augmentation only      : {aug_only}")
    print(f"  neither                : {neither}")
    print(f"  discordant pairs       : {discordant}")
    print(f"  accuracy difference    : {(aug_correct.mean() - gap_correct.mean()):+.4f}")
    print(f"  exact two-sided p      : {test.pvalue:.3e}")
    print(f"  95% CI on P(aug wins | disagree): "
          f"{test.proportion_ci().low:.3f} - {test.proportion_ci().high:.3f}"
          "   (0.5 = no difference)")
    print("  the augmentation attack is significantly better ✓"
          if test.pvalue < 0.05 else
          "  not significant at p < 0.05 ✗")

    # For contrast: the unpaired two-proportion test, which is what you get by
    # treating the two accuracies as if they came from separate samples. It
    # ignores that 4707 of 5000 points are scored identically by both attacks,
    # and pays for it with a p-value many orders of magnitude weaker.
    p_aug, p_gap = aug_correct.mean(), gap_correct.mean()
    n_test = len(y_true)
    unpaired_se = np.sqrt(p_aug * (1 - p_aug) / n_test
                          + p_gap * (1 - p_gap) / n_test)
    print(f"  (unpaired two-proportion z = {(p_aug - p_gap) / unpaired_se:.2f}, "
          f"SE {unpaired_se:.4f} — discards the pairing)")
    return test


def main():
    results = load_results()
    describe_arrays(results)
    print_saved_scores(results)

    features, labels = build_design_matrix(results)
    train_idx, test_idx = load_split(results, n_points=len(features))
    classifier = refit_attack_classifier(features, labels, train_idx)
    check_reproduction(classifier, features, labels, test_idx,
                       expected_acc=float(results["aug_acc"]))

    names = query_names(results)
    summarise_coefficients(classifier, features, train_idx, names)
    refit_without_duplicates(features, labels, train_idx, test_idx, names)
    run_ablations(results, features, labels, train_idx, test_idx, names)
    mcnemar_test(classifier, features, labels, test_idx)


if __name__ == "__main__":
    main()

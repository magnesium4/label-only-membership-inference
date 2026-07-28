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
import numpy as np
from sklearn.linear_model import LogisticRegression

RESULTS_PATH = "phase2_results.npz"
MAX_ITER = 1000          # matches the Phase 2 fit; changing it changes the model
REFIT_TOLERANCE = 1e-12  # lbfgs is deterministic, so only float noise is allowed


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


def main():
    results = load_results()
    describe_arrays(results)
    print_saved_scores(results)

    features, labels = build_design_matrix(results)
    train_idx, test_idx = load_split(results, n_points=len(features))
    classifier = refit_attack_classifier(features, labels, train_idx)
    check_reproduction(classifier, features, labels, test_idx,
                       expected_acc=float(results["aug_acc"]))


if __name__ == "__main__":
    main()

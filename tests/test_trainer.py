import pandas as pd

from voltguard.diagnostics.trainer import _pinned_profiles, _profile_folds


def _toy():
    # 10 drives x 3 rows; class 2 only in drive 7, class 1 in drives 0-4.
    groups = pd.Series([g for g in range(10) for _ in range(3)])
    y = pd.Series([1 if g < 5 else 0 for g in groups])
    y[groups == 7] = 2
    return y, groups


def test_single_drive_class_is_pinned_and_not_evaluable():
    y, groups = _toy()
    pinned, rare = _pinned_profiles(y, groups)
    assert rare == [2]
    assert pinned == {7}


def test_profile_folds_never_split_a_drive_or_test_pinned():
    y, groups = _toy()
    pinned, _ = _pinned_profiles(y, groups)
    tested = []
    for train_idx, test_idx in _profile_folds(groups, pinned, n_folds=3, seed=0):
        train_drives = set(groups.iloc[train_idx])
        test_drives = set(groups.iloc[test_idx])
        assert train_drives.isdisjoint(test_drives)
        assert 7 in train_drives and 7 not in test_drives
        tested.extend(test_drives)
    # Every non-pinned drive is tested exactly once across folds.
    assert sorted(tested) == [g for g in range(10) if g != 7]

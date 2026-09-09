import numpy as np
import pytest

from sparc.scoring.scoring_signals import compute_scoring_signals


@pytest.mark.parametrize('raw_tracks', [{}, {'cotracker_tracks': None}, {'cotracker_tracks': []}])
def test_missing_tracks_preserve_candidate_geometry_and_zero_motion(raw_tracks):
    result = compute_scoring_signals(
        raw_tracks, np.array([0.8, 0.4]),
        np.array([[0, 0, 3, 4], [0, 0, 6, 8]]),
        [], None, None, 15.0, 1,
    )
    candidates = result['all_candidate_breakdowns']
    assert [row['bbox_diag'] for row in candidates] == [5.0, 10.0]
    assert [row['is_winner'] for row in candidates] == [False, True]
    assert all(row['phase_fallback'] for row in candidates)
    assert all(row['snr_score_raw'] == 0.0 for row in candidates)
    assert all(row['mean_movement'] == 0.0 for row in candidates)
    assert result['snr_score_raw'] == 0.0

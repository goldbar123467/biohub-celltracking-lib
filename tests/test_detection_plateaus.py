import numpy as np

from biohub_ct.detection.classical import detect_local_maxima_3d


def test_bright_peak_ranking_uses_original_intensity_above_clip_quantile():
    frame = np.ones((16, 16, 16), dtype=np.uint16)
    frame[4:8, 4:8, 4:8] = 200
    frame[5, 5, 5] = 1000
    frame[12, 12, 12] = 2000
    peaks = detect_local_maxima_3d(frame, max_peaks=2)
    assert [p[:3] for p in peaks] == [(12, 12, 12), (5, 5, 5)]

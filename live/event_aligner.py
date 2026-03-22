"""Compute event-aligned features from live tick data.

Mirrors the logic in pipline_modified.add_event_aligned_features() for
the 'bar_close' trigger_source case used by flow_reversal scanner.
"""
from __future__ import annotations
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

POST_WINDOW_S = 15


def compute_event_features(
    tick_buffer: deque[tuple[int, float, float]],
    bar_start_ns: int,
    event_level: float,
    event_dir: int,
    atr: float,
) -> dict:
    """Compute event-aligned features from tick buffer.

    Args:
        tick_buffer: deque of (ts_ns, price, size) tuples
        bar_start_ns: bar start timestamp in nanoseconds
        event_level: price at trigger (close of bar)
        event_dir: 1 for long, -1 for short
        atr: current ATR value

    Returns:
        dict with sec_in_bar, event_return, event_effort_vs_result,
        event_rejection_strength, time_to_reject_s
    """
    defaults = {
        "sec_in_bar": 59.0,
        "event_return": 0.0,
        "event_effort_vs_result": 0.0,
        "event_rejection_strength": 0.0,
        "time_to_reject_s": float(POST_WINDOW_S),
    }

    # Convert buffer to arrays for the relevant time window
    one_min_ns = int(60 * 1e9)
    bar_end_ns = bar_start_ns + one_min_ns
    post_ns = int(POST_WINDOW_S * 1e9)

    # Extract ticks as numpy arrays
    ticks = [(ts, px, sz) for ts, px, sz in tick_buffer if ts >= bar_start_ns]
    if not ticks:
        return defaults

    t_ns = np.array([t[0] for t in ticks], dtype=np.int64)
    price = np.array([t[1] for t in ticks], dtype=np.float64)
    size = np.array([t[2] for t in ticks], dtype=np.float64)

    # For bar_close trigger: t0 is the last trade in the bar
    j0 = np.searchsorted(t_ns, bar_start_ns, side='left')
    j1 = np.searchsorted(t_ns, bar_end_ns, side='right')
    if j0 >= j1:
        return defaults

    t0_pos = j1 - 1
    t0_ns = t_ns[t0_pos]
    break_price = price[t0_pos]

    sec_in_bar = (t0_ns - bar_start_ns) / 1e9
    defaults["sec_in_bar"] = sec_in_bar

    # Post-window: from t0 to t0 + post_window_s
    end_ns = t0_ns + post_ns
    k0 = np.searchsorted(t_ns, t0_ns, side='left')
    k1 = np.searchsorted(t_ns, end_ns, side='right')

    if k0 >= k1 - 1:
        return defaults

    p_w = price[k0 + 1: k1]
    s_w = size[k0 + 1: k1]

    if len(p_w) == 0:
        return defaults

    price_end = float(p_w[-1])
    vol_w = float(s_w.sum())
    max_p = float(p_w.max())
    min_p = float(p_w.min())

    # event_return
    defaults["event_return"] = (price_end - break_price) / (break_price + 1e-12)

    # event_effort_vs_result
    price_move = abs(price_end - break_price)
    defaults["event_effort_vs_result"] = np.log1p(vol_w) / (price_move + 1e-9)

    # event_rejection_strength
    atr_safe = atr if (np.isfinite(atr) and atr > 0) else 1.0
    direction = 'up' if event_dir > 0 else 'down'
    if direction == 'up':
        rejection_mag = max_p - price_end
        back_mask = p_w < event_level
    else:
        rejection_mag = price_end - min_p
        back_mask = p_w > event_level

    defaults["event_rejection_strength"] = rejection_mag / atr_safe

    # time_to_reject_s
    if back_mask.any():
        back_rel = int(np.argmax(back_mask))
        back_t_ns = t_ns[k0 + 1 + back_rel]
        diff_ns = back_t_ns - t0_ns
        defaults["time_to_reject_s"] = max(diff_ns / 1e9, 0.001)

    return defaults

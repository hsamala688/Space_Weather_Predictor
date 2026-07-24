"""Fixed scientific contract: grid, driver features, and step counts.

This module holds only values that are part of the *data contract* and do not
change between environments or runs. Environment-tunable parameters (data root,
date range, source URLs, split years, chunk size) live in ``settings.py``; every
on-disk path is resolved through ``settings.DataLayout``.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Grid contract (the SFNO transform's Gauss-Legendre grid)
# ---------------------------------------------------------------------------

LMAX = 22
NLAT = 23
NLON = 45
AALT = np.arange(80, 2001, 20).astype(float)  # IRI integration altitudes, km

# ---------------------------------------------------------------------------
# Driver contract
# ---------------------------------------------------------------------------

OMNI_HRO_FEATURES = ["b_magnitude", "by_gsm", "bz_gsm", "flow_speed", "proton_density"]
KP_FEATURE = "kp_3hour"
DRIVER_FEATURES = OMNI_HRO_FEATURES + [KP_FEATURE]
INPUT_STEPS = 6
TARGET_STEPS = 3

# Daily F10.7 above this is a burst-inflated single-day spike, not a valid EUV
# proxy: set to NaN and time-interpolate (do NOT clip).
F107_MAX = 300.0

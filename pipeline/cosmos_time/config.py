"""Every tunable number in the pipeline, in one place.

Nothing here is tuned to make a validation check pass. Selection cuts follow
the plan agreed before any data was fetched; the integrator step is fixed by
the convergence test in tests/test_engineering.py, not by the checks.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # pipeline/
REPO = ROOT.parent
CACHE = ROOT / 'cache'                               # raw query results (gitignored)
REPORTS = ROOT / 'reports'                           # validation output (committed)
OUT = REPO / 'time' / 'data'                         # files the page loads

# --------------------------------------------------------------- units ----
KMS_TO_PCMYR = 1.0227121650537077                    # 1 km/s in pc/Myr
PC_PER_LY = 0.30660139378555057
MAS_YR_TO_KMS_PER_PC = 4.740470463533348             # v_t = 4.74 * mu[as/yr] * d[pc]

# ----------------------------------------------- reference epoch & frame ----
EPOCH = 2016.0                                       # Gaia DR3 epoch; t = 0 in the app
HIP_EPOCH = 1991.25
# astropy Galactocentric frame, 'v4.0' defaults (recorded in the manifest)
FRAME = 'v4.0'
R0_KPC = 8.122                                       # GRAVITY 2018
Z_SUN_PC = 20.8                                      # Bennett & Bovy 2019
V_SUN_KMS = (12.9, 245.6, 7.78)                      # Drimmel & Poggio 2018

# ------------------------------------------------------------- selection ----
# Local sample: nearest first. parallax > 4.4 mas (~227 pc) gives ~800k stars
# after the quality cuts (count queries run 2026-10-02: >4.5 mas → 766,330;
# >4.0 mas → 969,256).
LOCAL_MIN_PARALLAX = 4.4
LOCAL_MIN_POE = 10.0
MAX_RUWE = 1.4
# Disk sample: a random draw of the full 6D catalogue beyond the local cut,
# so the view from above the disk is a real patch of galaxy, not a dot.
DISK_MIN_POE = 5.0
DISK_TARGET = 200_000
# Bright stars: every Hipparcos star brighter than this V uses Gaia only if its
# Gaia 6D solution passes the cuts above and G > GAIA_BRIGHT_LIMIT; otherwise
# Hipparcos (van Leeuwen 2007) astrometry + XHIP radial velocity.
NAKED_EYE_V = 6.5
GAIA_BRIGHT_LIMIT = 3.0

# ------------------------------------------------------------- potential ----
POTENTIAL = 'McMillan17'                             # galpy.potential.mwpotentials
GRID_N = 512                                         # table is GRID_N x GRID_N
GRID_A_R = 1000.0                                    # pc; u = asinh(R / A_R)
GRID_A_Z = 50.0                                      # pc; v = asinh(|z| / A_Z)
GRID_R_MAX = 120_000.0                               # pc
GRID_Z_MAX = 120_000.0                               # pc

# ------------------------------------------------------------ integrator ----
DT_MYR = 0.1                                         # fixed leapfrog step
T_MAX_MYR = 250.0                                    # range each way

# --------------------------------------------------------------- packing ----
FIRST_TIER = 100_000                                 # brightest-from-Earth first
TIER_SIZE = 250_000
VEL_QUANT = 0.01                                     # km/s per int16 step
DIST_LOG_MIN, DIST_LOG_MAX = 0.0, 4.5                # log10(pc) range of uint16 distance
PRECISION_RADIUS_PC = 25.0                           # inside this, store full float32
ID_SHARD = 32_768                                    # stars per lazily-loaded ids/info file

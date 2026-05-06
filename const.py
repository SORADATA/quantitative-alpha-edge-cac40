from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / 'src' / 'models'
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"
DEFAULT_MARKET = "cac40.json"
CONFIG_FILE = BASE_DIR / DEFAULT_MARKET

# =============================================================================
# MARKET & RISK
# =============================================================================

TRADING_DAYS_YEAR: int = 252
RISK_FREE_RATE:    float = 0.03

# =============================================================================
# ML SIGNAL
# =============================================================================

TARGET_CLUSTER:   int = 3
PROBA_THRESHOLD:  float = 0.6

FEATURE_COLS: list[str] = [
    "rsi",
    "macd",
    "bb_low",
    "bb_high",
    "atr",
    "return_2m",
    "return_3m",
    "return_6m",
    "euro_volume_lag1",
    "garman_klass_vol_lag1",
    "Mkt-RF_lag1",
    "SMB_lag1",
    "HML_lag1",
    "RMW_lag1",
    "CMA_lag1",
    "cluster",
]

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

RSI_WINDOW:      int = 20
BB_WINDOW:       int = 20
BB_STD:          int = 2
ATR_WINDOW:      int = 14
MACD_SLOW:       int = 26
MACD_FAST:       int = 12
MACD_SIGN:       int = 9

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

MIN_HISTORY_TA:  int = 20   # minimum bars for RSI / Bollinger
MIN_HISTORY_FF:  int = 24   # minimum bars for Fama-French rolling betas

MOMENTUM_LAGS: list[int] = [1, 2, 3, 6, 9, 12]
WINSOR_CUTOFF: float = 0.005

VARS_TO_LAG: list[str] = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
    "euro_volume",
    "garman_klass_vol",
]

FAMA_FRENCH_FACTORS: list[str] = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

# =============================================================================
# RESAMPLING
# =============================================================================

RESAMPLE_MEAN_COLS: list[str] = ["euro_volume"]

RESAMPLE_LAST_EXCLUDE: list[str] = [
    "euro_volume",
    "volume",
    "open",
    "high",
    "low",
    "close",
]


from pathlib import Path
from dotenv import load_dotenv
import os


# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "src" / "models"
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "configs"

load_dotenv(BASE_DIR / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")


# =============================================================================
# MARKET & RISK
# =============================================================================
TRADING_DAYS_YEAR: int = 252
RISK_FREE_RATE:    float = 0.03


# =============================================================================
# ML SIGNAL — paramètres de sélection
# =============================================================================
# Utilisés dans backtest.py (_select_tickers) et dans les configs par marché.
# TARGET_CLUSTER et PROBA_THRESHOLD sont gardés pour rétrocompatibilité
# mais PROBA_MIN est la référence du nouveau backtest.
TARGET_CLUSTER:   int = 3
PROBA_THRESHOLD:  float = 0.60
PROBA_MIN:        float = 0.55
MAX_STOCKS:       int = 15     # nombre max de titres en portefeuille


# =============================================================================
# FEATURE COLUMNS
# =============================================================================
# ⚠️  Règle critique : toutes les features calculées sur T doivent être laggées
#     d'une période (→ _lag1) pour éviter le data leakage.
#
#     Exceptions (calculées sur T-1 par construction) :
#       - return_Nm  : pct_change(N) sur adj close → déjà du passé
#       - Fama-French _lag1 : laggés dans processor.py (VARS_TO_LAG)
#       - alpha features     : calculées sur rendements passés
#
#     Indicateurs TA (rsi, macd, bb_*, atr, cluster) :
#       → Calculés sur le prix de clôture du mois T
#       → DOIVENT être laggés : utiliser les versions _lag1

FEATURE_COLS: list[str] = [
    # ── Indicateurs techniques (laggés — anti-leakage)
    "rsi_lag1",
    "macd_lag1",
    "bb_low_lag1",
    "bb_mid_lag1",
    "bb_high_lag1",
    "atr_lag1",
    "cluster_lag1",

    # ── Momentum (rendements passés — pas de leakage)
    "return_1m",
    "return_2m",
    "return_3m",
    "return_6m",
    "return_9m",
    "return_12m",

    # ── Alpha features momentum (skip-1m)
    "mom_12_1",
    "mom_6_1",

    # ── Volatilité
    "realized_vol_3m",
    "realized_vol_12m",
    "vol_ratio",

    # ── Risk-adjusted
    "sharpe_3m",
    "sharpe_6m",
    "sortino_6m",

    # ── Tail risk
    "return_skew_6m",
    "hist_var_5pct",
    "cvar_5pct",

    # ── Liquidité
    "amihud_illiquidity",
    "volume_zscore",

    # ── Mean reversion
    "price_zscore_12",
    "nearness_52w_high",

    # ── Features cross-sectionnelles (rank percentile)
    "mom_12_1_rank",
    "sharpe_6m_rank",
    "realized_vol_3m_rank",
    "amihud_illiquidity_rank",

    # ── Macro Fama-French (laggés dans VARS_TO_LAG)
    "Mkt-RF_lag1",
    "SMB_lag1",
    "HML_lag1",
    "RMW_lag1",
    "CMA_lag1",

    # ── Macro agrégée (laggée dans VARS_TO_LAG)
    "euro_volume_lag1",
    "garman_klass_vol_lag1",

    # ── Saisonnalité
    "month_sin",
    "month_cos",
    "is_q_end",
]


# =============================================================================
# TECHNICAL INDICATORS — fenêtres
# =============================================================================
RSI_WINDOW: int = 20
BB_WINDOW:  int = 20
BB_STD:     int = 2
ATR_WINDOW: int = 14
MACD_SLOW:  int = 26
MACD_FAST:  int = 12
MACD_SIGN:  int = 9


# =============================================================================
# FEATURE ENGINEERING — historique minimum
# =============================================================================
MIN_HISTORY_TA: int = 20     # jours minimum pour calculer les indicateurs TA
MIN_HISTORY_FF: int = 24     # mois minimum pour le rolling OLS Fama-French
WINSOR_CUTOFF:  float = 0.005  # cutoff winsorisation (0.5% / 99.5%)

MOMENTUM_LAGS: list[int] = [1, 2, 3, 6, 9, 12]   # lags pour calculate_returns()


# =============================================================================
# LAGS — anti-leakage
# =============================================================================
VARS_TO_LAG: list[str] = [
    # Facteurs Fama-French
    "Mkt-RF", "SMB", "HML", "RMW", "CMA",
    # Volume & volatilité agrégée
    "euro_volume",
    "garman_klass_vol",
]

FAMA_FRENCH_FACTORS: list[str] = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


# =============================================================================
# RESAMPLING — agrégation daily → monthly
# =============================================================================
# Colonnes agrégées par MOYENNE mensuelle (flux)
RESAMPLE_MEAN_COLS: list[str] = ["euro_volume"]

RESAMPLE_LAST_EXCLUDE: list[str] = [
    "euro_volume",
    "volume",
    "open",
    "high",
    "low",
    "close",
]

SHARPE_THRESHOLD = 0.5
MAX_DD_THRESHOLD = -0.15


TRANSACTION_COST = 0.0010
MIN_STOCKS_OPTIM = 3
MAX_STOCKS_SELECT = 15
PROBA_MIN = 0.55
WEIGHT_BOUNDS = (0.02, 0.20)


FEATURE_GROUPS = {
    "momentum": [
        "return_1m", "return_2m", "return_3m", "return_6m", "return_9m", "return_12m",
        "mom_12_1", "mom_6_1", "mom_3_1", "mom_12_1_rank",
    ],
    "volatility": [
        "realized_vol_3m", "realized_vol_12m", "vol_ratio",
        "realized_vol_3m_rank", "garman_klass_vol_lag1", "idio_vol",
    ],
    "risk_adjusted": [
        "sharpe_3m", "sharpe_6m", "sortino_6m", "calmar_proxy", "sharpe_6m_rank",
    ],
    "tail_risk": [
        "return_skew_6m", "return_kurt_6m", "hist_var_5pct", "cvar_5pct",
    ],
    "technical": [
        "rsi_lag1", "macd_lag1", "bb_low_lag1", "bb_mid_lag1",
        "bb_high_lag1", "atr_lag1", "cluster_lag1",
        "bb_position", "rsi_divergence", "macd_sign",
    ],
    "liquidity": [
        "amihud_illiquidity", "volume_trend_3m", "volume_zscore",
        "amihud_illiquidity_rank", "euro_volume_lag1",
    ],
    "mean_reversion": [
        "price_zscore_12", "nearness_52w_high",
    ],
    "macro": [
        "Mkt-RF_lag1", "SMB_lag1", "HML_lag1", "RMW_lag1", "CMA_lag1",
    ],
    "seasonality": [
        "month_sin", "month_cos", "is_q_end", "is_jan",
    ],
}

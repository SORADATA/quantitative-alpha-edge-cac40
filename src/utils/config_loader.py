import json
import os
from typing import Dict, Any, List
from src.utils.logger import setup_logger
from const import CONFIG_DIR

logger = setup_logger("ConfigLoader")


def load_market_config(config_name: str) -> Dict[str, Any]:
    """
    Charge une configuration de marché spécifique depuis un fichier JSON.
    Par défaut, tente de charger le CAC 40.
    """
    target_path = CONFIG_DIR / config_name

    # Valeurs de secours (au cas où le fichier JSON est corrompu ou absent)
    default_fallback = {
        "market_name": "CAC 40 (France)",
        "benchmark_ticker": "^FCHI",
        "currency": "EUR",
        "assets": ["AI.PA", "AIR.PA", "OR.PA", "MC.PA", "SAN.PA"]
    }

    if not target_path.exists():
        logger.warning(
            f"Fichier {config_name} introuvable dans {CONFIG_DIR}. Utilisation du fallback."
            )

        return default_fallback

    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            logger.info(f"Configuration chargée : {config_name}")
            return config_data
    except json.JSONDecodeError as e:
        logger.error(f"Erreur de syntaxe JSON dans {target_path} : {e}")
        return default_fallback
    except Exception as e:
        logger.error(f"Erreur imprévue lors du chargement de {target_path} : {e}")
        return default_fallback


def get_available_configs() -> List[str]:
    """Liste tous les fichiers .json disponibles pour le menu de l'App."""
    return [f.name for f in CONFIG_DIR.glob("*.json")]

# =============================================================================
# INITIALISATION AUTOMATIQUE
# =============================================================================

# 1. On donne la priorité au CAC 40 par défaut
# La variable d'env permet de changer sans toucher au code (ex: pour GitHub Actions)


selected_market_file = os.getenv("MARKET_TYPE", "cac40.json")

# 2. Chargement des données
MARKET_CONFIG = load_market_config(selected_market_file)

TICKERS: List[str] = MARKET_CONFIG.get('assets', [])
BENCHMARK_TICKER: str = MARKET_CONFIG.get('benchmark_ticker', '^FCHI')
MARKET_NAME: str = MARKET_CONFIG.get('market_name', 'CAC 40')
CURRENCY: str = MARKET_CONFIG.get('currency', 'EUR')

logger.info(f"AlphaEdge prêt pour {MARKET_NAME} | {len(TICKERS)} tickers | Devise: {CURRENCY}")
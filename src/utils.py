import argparse
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import requests
import yaml


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(name)


def load_settings(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def parse_args(default_config: str = "config/settings.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Molecular informatics pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def fetch_pubchem_properties(
    identifier: str,
    id_type: str,
    base_url: str,
    max_retries: int,
    retry_backoff: int,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    properties = "CanonicalSMILES,InChI,MolecularWeight,XLogP3,TPSA,HBondDonorCount,HBondAcceptorCount"
    url = f"{base_url}/compound/{id_type}/{requests.utils.quote(identifier)}/property/{properties}/JSON"
    attempt = 0
    while attempt <= max_retries:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                payload = response.json()
                props = payload.get("PropertyTable", {}).get("Properties", [])
                if not props:
                    logger.warning("No properties returned for %s", identifier)
                    return None
                record = props[0]
                record["query"] = identifier
                return record
            if response.status_code == 404:
                logger.error("Identifier %s not found in PubChem", identifier)
                return None
            logger.warning(
                "Attempt %d for %s failed with status %s", attempt + 1, identifier, response.status_code
            )
        except requests.RequestException as exc:
            logger.error("Request error for %s: %s", identifier, exc)
        attempt += 1
        time.sleep(retry_backoff * attempt)
    logger.error("Max retries exceeded for %s", identifier)
    return None


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


__all__ = [
    "get_logger",
    "load_settings",
    "ensure_directories",
    "parse_args",
    "fetch_pubchem_properties",
    "save_json",
]

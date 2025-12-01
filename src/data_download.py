import os
from typing import List

import pandas as pd

from . import utils


def fetch_compounds(config_path: str) -> pd.DataFrame:
    settings = utils.load_settings(config_path)
    logger = utils.get_logger(__name__)

    pubchem_cfg = settings.get("pubchem", {})
    base_url = pubchem_cfg.get("base_url")
    identifiers: List[str] = pubchem_cfg.get("identifiers", [])
    id_type = pubchem_cfg.get("id_type", "name")
    max_retries = int(pubchem_cfg.get("max_retries", 3))
    retry_backoff = int(pubchem_cfg.get("retry_backoff", 2))

    if not base_url:
        raise ValueError("PubChem base_url is not configured")

    if not identifiers:
        raise ValueError("No identifiers provided in configuration")

    logger.info("Fetching %d compounds from PubChem (%s)", len(identifiers), id_type)

    records = []
    for identifier in identifiers:
        record = utils.fetch_pubchem_properties(
            identifier=identifier,
            id_type=id_type,
            base_url=base_url,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            logger=logger,
        )
        if record:
            records.append(record)

    if not records:
        raise RuntimeError("No records retrieved from PubChem")

    df = pd.DataFrame(records)
    logger.info("Retrieved %d records", len(df))
    return df


def main() -> None:
    args = utils.parse_args()
    settings = utils.load_settings(args.config)
    logger = utils.get_logger(__name__)

    raw_path = settings.get("paths", {}).get("raw_data", "data/raw/compounds.csv")
    utils.ensure_directories([os.path.dirname(raw_path)])

    df = fetch_compounds(args.config)
    df.to_csv(raw_path, index=False)
    logger.info("Saved raw data to %s", raw_path)


if __name__ == "__main__":
    main()

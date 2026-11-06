"""Run the full LoomIQ training pipeline in order.

Steps:
    1. Clean the raw data and build the feature table.
    2. Train the productivity model (CV -> refit best -> test -> save).
    3. Save SHAP explanation figures.
    4. Prepare the fabric tiles and train the defect model (skipped if AITEX
       has not been downloaded yet).

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --skip-vision
"""

import argparse
from collections.abc import Callable

from loomiq.data.clean import run_cleaning
from loomiq.features.build_features import run_feature_building
from loomiq.models.explain import save_global_explanations
from loomiq.models.train_productivity import train_productivity_model
from loomiq.utils.logger import get_logger

logger = get_logger("train_all")


def run_step(name: str, func: Callable[[], object], optional: bool = False) -> bool:
    """Run one pipeline step. Optional steps are skipped (not failed) when their data is missing."""
    logger.info("Starting step: %s", name)
    try:
        func()
    except FileNotFoundError as error:
        if not optional:
            raise
        logger.warning("Skipped '%s': %s", name, error)
        return False
    logger.info("Finished step: %s", name)
    return True


def train_vision() -> None:
    from loomiq.vision.train_defect import train_defect_model

    train_defect_model()


def main() -> None:
    """Parse arguments and run each training step."""
    parser = argparse.ArgumentParser(description="Train all LoomIQ models.")
    parser.add_argument("--skip-vision", action="store_true", help="Skip the defect model.")
    args = parser.parse_args()

    run_step("clean data", run_cleaning)
    run_step("build features", run_feature_building)
    run_step("productivity model", train_productivity_model)
    run_step("SHAP explanations", save_global_explanations)
    if not args.skip_vision:
        run_step("defect model", train_vision, optional=True)


if __name__ == "__main__":
    main()

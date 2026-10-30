"""Run the full LoomIQ training pipeline in order.

Steps:
    1. Clean the raw data and build the feature table.
    2. Train the productivity model (CV -> refit best -> test -> save).
    3. Save SHAP explanation figures.
    4. Prepare the fabric defect dataset.
    5. Train the defect model.

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
from loomiq.vision.prepare_dataset import prepare_fabric_dataset
from loomiq.vision.train_defect import train_defect_model

logger = get_logger("train_all")


def run_step(name: str, func: Callable[[], object]) -> bool:
    """Run one pipeline step and report whether it succeeded."""
    logger.info("Starting step: %s", name)
    try:
        func()
    except NotImplementedError as error:
        logger.warning("Skipped '%s' (not implemented yet): %s", name, error)
        return False
    logger.info("Finished step: %s", name)
    return True


def main() -> None:
    """Parse arguments and run each training step."""
    parser = argparse.ArgumentParser(description="Train all LoomIQ models.")
    parser.add_argument("--skip-vision", action="store_true", help="Skip defect model steps.")
    args = parser.parse_args()

    run_step("clean data", run_cleaning)
    run_step("build features", run_feature_building)
    run_step("productivity model", train_productivity_model)
    run_step("SHAP explanations", save_global_explanations)
    if not args.skip_vision:
        run_step("prepare fabric dataset", prepare_fabric_dataset)
        run_step("defect model", train_defect_model)


if __name__ == "__main__":
    main()

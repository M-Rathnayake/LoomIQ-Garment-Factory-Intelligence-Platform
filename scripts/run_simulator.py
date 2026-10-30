"""Run the factory simulator and stream hourly data to a CSV file.

The dashboard reads this file to show live line status.

Usage:
    python scripts/run_simulator.py
    python scripts/run_simulator.py --lines 8 --hours 10
"""

import argparse

from loomiq.config import load_config
from loomiq.simulation.factory_simulator import FactorySimulator
from loomiq.utils.logger import get_logger

logger = get_logger("run_simulator")


def main() -> None:
    """Parse arguments and start the simulator."""
    sim_config = load_config()["simulation"]

    parser = argparse.ArgumentParser(description="Stream simulated factory data.")
    parser.add_argument("--lines", type=int, default=sim_config["n_lines"])
    parser.add_argument("--hours", type=int, default=sim_config["hours_per_shift"])
    args = parser.parse_args()

    simulator = FactorySimulator(n_lines=args.lines, hours_per_shift=args.hours)
    logger.info("Simulating %s lines for %s hours", args.lines, args.hours)

    try:
        for hourly_df in simulator.stream(hours=args.hours):
            # TODO: Append hourly_df to sim_config["output_file"] (create header once).
            # TODO: time.sleep(sim_config["seconds_between_ticks"]) to mimic real time.
            logger.info("Generated %s rows", len(hourly_df))
    except NotImplementedError as error:
        logger.warning("Simulator not implemented yet: %s", error)


if __name__ == "__main__":
    main()

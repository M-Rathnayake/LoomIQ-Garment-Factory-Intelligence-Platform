"""Run the factory simulator from the command line and save the hourly stream to CSV.

The dashboard runs its own simulator in the browser, so this script is optional.
It is useful to inspect the simulated data, or to feed it into other tools.

Usage:
    python scripts/run_simulator.py
    python scripts/run_simulator.py --lines 8 --seed 3 --realtime
"""

import argparse
import time

from loomiq.config import PROJECT_ROOT, load_config
from loomiq.simulation.factory_simulator import FactorySimulator
from loomiq.utils.logger import get_logger

logger = get_logger("run_simulator")


def main() -> None:
    """Parse arguments, simulate one shift and write every hour to CSV."""
    sim_config = load_config()["simulation"]

    parser = argparse.ArgumentParser(description="Stream one simulated factory day to CSV.")
    parser.add_argument("--lines", type=int, default=sim_config["n_lines"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date", default=None, help="replay this date (YYYY-MM-DD) instead of a random test day")
    parser.add_argument("--realtime", action="store_true", help="pause between hours, like a live feed")
    args = parser.parse_args()

    simulator = FactorySimulator(n_lines=args.lines, seed=args.seed, date=args.date)
    output = PROJECT_ROOT / sim_config["output_file"]
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Replaying %s with %s lines -> %s", simulator.date.date(), len(simulator.lines), output)

    for hour, hourly_df in enumerate(simulator.stream(), start=1):
        hourly_df.to_csv(output, mode="w" if hour == 1 else "a", header=hour == 1, index=False)
        behind = (hourly_df["pieces_so_far"] < hourly_df["target_so_far"]).sum()
        logger.info("Hour %s: %s pieces, %s of %s lines behind target pace",
                    hour, f"{hourly_df['pieces_this_hour'].sum():,.0f}", behind, len(hourly_df))
        if args.realtime:
            time.sleep(sim_config["seconds_between_ticks"])


if __name__ == "__main__":
    main()

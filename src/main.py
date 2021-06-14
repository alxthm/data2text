import datetime
import logging
import sys

from src.train import train
from src.utils import (
    WarningsFilter,
)

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="[%H:%M:%S]", level=logging.INFO
)


if __name__ == "__main__":
    # filter out some hdfs warnings (from 3rd party python libraries)
    sys.stdout = WarningsFilter(sys.stdout)
    sys.stderr = WarningsFilter(sys.stderr)
    timestamp = datetime.datetime.today().strftime("%m%d%H%M%S")
    train(timestamp=timestamp)

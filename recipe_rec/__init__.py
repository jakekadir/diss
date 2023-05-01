import logging

RANDOM_STATE: int = 42

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s.%(module)s: %(message)s", datefmt="%H:%M:%S"
)

import logging
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-log",
    "--loglevel",
    default="warning",
    help="Provide logging level. Example --loglevel debug, default=warning",
)
args = parser.parse_args()
logging.basicConfig(level=args.loglevel.upper())
logging.info(f"Logging level {args.loglevel.upper()}.")
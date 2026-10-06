import sys

sys.dont_write_bytecode = True

from model.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

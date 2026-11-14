#!/usr/bin/env python3

import importlib
import sys


COMMANDS = {
    "optimize": "mfrlcd.cli",
    "batch": "mfrlcd.batch",
    "doctor": "mfrlcd.runtime",
    "export": "mfrlcd.results",
    "summarize": "mfrlcd.results",
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments and arguments[0] in COMMANDS else "optimize"
    if arguments and arguments[0] in COMMANDS:
        arguments.pop(0)
    module = importlib.import_module(COMMANDS[command])
    if command in {"export", "summarize"}:
        return module.main(arguments, command=command)
    return module.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

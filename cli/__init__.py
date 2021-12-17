from argparse import ArgumentParser
from typing import Dict


def create_parser(sub_commands: Dict[str, ArgumentParser]) -> ArgumentParser:
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='command')

    for name, sub_parser in sub_commands.items():
        sub_parsers.add_parser(name=name, parents=[sub_parser])

    return parser

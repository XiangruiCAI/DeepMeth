from cli import create_parser
from cli.parsers import preprocess_parser

sub_commands = {
    'preprocess': preprocess_parser()
}

parser = create_parser(sub_commands)

args = parser.parse_args()
args.func(args)

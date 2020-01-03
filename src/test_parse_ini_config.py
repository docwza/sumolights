from unittest import TestCase
from src.parse_ini_config import get_config
from src.argparse import parse_cl_args, update_args


class Test(TestCase):
    def test_get_config(self):
        sample_config_filepath = "../config/sample.ini"
        config_parser = get_config(sample_config_filepath)
        args = parse_cl_args()
        args = update_args(args, config_parser)
        assert (args.n == 6)
        assert (args.l == 1)
        assert (args.nogui == True)

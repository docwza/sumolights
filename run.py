import os, sys, time
import warnings
from src.parse_ini_config import get_config
warnings.filterwarnings('ignore')
from src.argparse import parse_cl_args, update_args
from src.distprocs import DistProcs


def main():
    start_t = time.time()
    print('start running main...')
    args = parse_cl_args()
    args = update_args(args, get_config("config/single_maxpressure.ini"))
    distprocs = DistProcs(args, args.tsc, args.mode)
    distprocs.run()
    print(args)
    print('...finish running main')
    print('run time '+str((time.time()-start_t)/60))


if __name__ == '__main__':
    main()

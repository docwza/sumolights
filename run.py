import os, sys, time

from src.argparse import parse_cl_args
from src.distprocs import DistProcs

def main():
    start_t = time.time()
    print('start running main...')
    args = parse_cl_args()
    distprocs = DistProcs(args, args.tsc, args.mode)
    distprocs.run()
    print(args)
    print('...finish running main')
    print('run time '+str((time.time()-start_t)/60))

if __name__ == '__main__':
    main()

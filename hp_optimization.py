import itertools, time, os, argparse, shutil

import numpy as np

from src.picklefuncs import load_data, save_data
from src.helper_funcs import check_and_make_dir, write_lines_to_file, write_line_to_file, get_time_now

def parse_cl_args():
    parser = argparse.ArgumentParser()

    #multi proc params
    parser.add_argument("-n", type=int, default=os.cpu_count()-1, dest='n', help='number of sim procs (parallel simulations) generating experiences, default: os.cpu_count()-1')
    parser.add_argument("-l", type=int, default=1, dest='l', help='number of parallel learner procs producing updates, default: 1')

    ##sumo params
    parser.add_argument("-sim", type=str, default='single', dest='sim', help='simulation scenario, default: lust, options:lust, single, double')
    parser.add_argument("-tsc", type=str, default='websters', dest='tsc', help='traffic signal control algorithm, default:websters; options:sotl, maxpressure, dqn, ddpg'  )

    args = parser.parse_args()
    return args

def get_hp_dict(tsc_str):
    if tsc_str == 'dqn':
        return {'-lr':[ 0.0001, 0.00005], '-lre':[0.0001, 0.000001, 0.0000001, 0.00000001], '-updates':[15000], '-batch':[32], '-nreplay':[15000], '-nsteps':[1,2], '-n_hidden':[3], '-target_freq':[32,64,128], '-gmin':[5,10]}
    elif tsc_str == 'ddpg':
        return {'-lr':[ 0.0001, 0.00005], '-lre':[0.0001, 0.000001, 0.00000001], '-updates':[15000], '-batch':[32], '-nreplay':[15000], '-tau':[0.01, 0.005], '-nsteps':[1], '-lrc':[ 0.0005, 0.0001], '-n_hidden':[3], '-target_freq':[1,16,64]}
    elif tsc_str == 'sotl':
        return {'-theta':[10,20,30,40,50], '-mu':[0,5,10,15], '-omega':[0,5,10,15]}
    elif tsc_str == 'websters':
        return {'-cmin':[40, 60, 80], '-cmax':[160, 180, 200], '-satflow':[0.3, 0.38, 0.44], '-f':[600, 900, 1800]}
    elif tsc_str == 'maxpressure':
        return {'-gmin':np.arange(5,26)}
    elif tsc_str == 'uniform': 
        return {'-gmin':np.arange(5,26)}
    else:
        #raise not found exceptions
        assert 0, 'Error: Supplied traffic signal control argument type '+str(tsc_str)+' does not exist.'

def create_hp_cmds(args, hp_order, hp):
    hp_cmds = []
    cmd_str = 'python run.py -sim '+str(args.sim)+' -nogui -tsc '+str(args.tsc)

    #create hp string
    hp_str = ' '
    for s, v in zip(hp_order, hp):
        hp_str += str(s)+' '+str(v)+' '

    if args.tsc == 'ddpg' or args.tsc == 'dqn':
        #train cmd for rl tsc
        #need to learn before can evaluate hp
        hp_cmds.append(cmd_str+hp_str+'-mode train -save -n '+str(args.n)+' -l '+str(args.l))

    #test cmd runs 'n' sims to gen results
    hp_cmds.append(cmd_str+hp_str+'-mode test -n '+str(args.n+args.l))
    #test cmd for rl tsc needs to load saved/learned weights
    if args.tsc == 'ddpg' or args.tsc == 'dqn':
        hp_cmds[-1] += ' -load'

    return hp_cmds

def get_hp_results(fp):
    travel_times = []
    for f in os.listdir(fp):
        travel_times.extend(load_data(fp+f))

    return travel_times

def rank_hp(hp_fitness, hp_order, tsc_str, fp):
    #fitness is the mean+std of the travel time
    ranked_hp_fitness = [ (hp, hp_fitness[hp]['mean']+hp_fitness[hp]['std']) for hp in hp_fitness]
    ranked_hp_fitness = sorted(ranked_hp_fitness, key=lambda x:x[-1]) 
    print('Best hyperparams set for '+str(tsc_str))
    print(hp_order)
    print(ranked_hp_fitness[0])

    #write all hps to file
    #write header line
    lines = [','.join(hp_order)+',mean,std,mean+std']
    #rest of lines are ranked hyperparams
    for hp in ranked_hp_fitness:
        hp_str = hp[0]
        lines.append( hp_str+','+str(hp_fitness[hp_str]['mean'])+','+str(hp_fitness[hp_str]['std'])+','+str(hp[1]))
    
    write_lines_to_file(fp, 'a+', lines)

def write_temp_hp(hp, results, fp):
    write_line_to_file(fp, 'a+', hp+','+str(results['mean'])+','+str(results['std'])+','+str(results['mean']+results['std']))

def save_hp_performance(data, path, hp_str): 
    check_and_make_dir(path)
    #name the return the unique hp string
    save_data(path+hp_str+'.p', data)

def main():
    start = time.time()

    args = parse_cl_args()

    #get hyperparams for command line supplied tsc
    tsc_str = args.tsc
    hp_dict = get_hp_dict(tsc_str)
    hp_order = sorted(list(hp_dict.keys()))

    hp_list = [hp_dict[hp] for hp in hp_order]
    #use itertools to produce cartesian product of hyperparams
    hp_set = list(itertools.product(*hp_list))
    print(str(len(hp_set))+' total hyper params')

    #where to find metrics
    hp_travel_times = {}
    metrics_fp = 'metrics/'+tsc_str

    #where to print hp results
    path = 'hyperparams/'+tsc_str+'/'
    check_and_make_dir(path)
    fname = get_time_now()
    hp_fp = path+fname+'.csv'
    write_line_to_file(hp_fp, 'a+', ','.join(hp_order)+',mean,std,mean+std' )

    #run each set of hp from cartesian product
    for hp in hp_set:                                                                                                                                                                                                    
        hp_cmds = create_hp_cmds(args, hp_order, hp)
        #print(hp_cmds)
        for cmd in hp_cmds:
            os.system(cmd)
        #read travel times, store mean and std for determining best hp set
        hp_str = ','.join([str(h) for h in hp])
        travel_times = get_hp_results(metrics_fp+'/traveltime/')
        hp_travel_times[hp_str] = {'mean':int(np.mean(travel_times)), 'std':int(np.std(travel_times))}
        write_temp_hp(hp_str, hp_travel_times[hp_str], hp_fp)
        #generate_returns(tsc_str, 'metrics/', hp_str)
        save_hp_performance(travel_times, 'hp/'+tsc_str+'/', hp_str) 
        #remove all metrics for most recent hp
        shutil.rmtree(metrics_fp)

    #remove temp hp and write ranked final results
    os.remove(hp_fp)
    rank_hp(hp_travel_times, hp_order, tsc_str, hp_fp)
    print('All hyperparamers performance can be viewed at: '+str(hp_fp))

    print('TOTAL HP SEARCH TIME')
    secs = time.time()-start
    print(str(int(secs/60.0))+' minutes ')

if __name__ == '__main__':
    main()

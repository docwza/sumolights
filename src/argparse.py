import argparse, os

def parse_cl_args():
    parser = argparse.ArgumentParser()

    #multi proc params
    parser.add_argument("-n", type=int, default=os.cpu_count()-1, dest='n', help='number of sim procs (parallel simulations) generating experiences, default: os.cpu_count()-1')
    parser.add_argument("-l", type=int, default=1, dest='l', help='number of parallel learner procs producing updates, default: 1')

    ##sumo params
    parser.add_argument("-sim", type=str, default=None, dest='sim', help='simulation scenario, default: lust, options:lust, single, double')
    parser.add_argument("-port", type=int, default=9000, dest='port', help='port to connect self.conn.server, default: 9000')
    parser.add_argument("-netfp", type=str, default='networks/double.net.xml', dest='net_fp', help='path to desired simulation network file, default: networks/double.net.xml')
    parser.add_argument("-sumocfg", type=str, default='networks/double.sumocfg', dest='cfg_fp', help='path to desired simulation configuration file, default: networks/double.sumocfg' )
    parser.add_argument("-mode", type=str, default='train', dest='mode', help='reinforcement mode, train (agents receive updates) or test (no updates), default:train, options: train, test'  )
    parser.add_argument("-tsc", type=str, default='websters', dest='tsc', help='traffic signal control algorithm, default:websters; options:sotl, maxpressure, dqn, ddpg'  )
    parser.add_argument("-simlen", type=int, default=10800, dest='sim_len', help='length of simulation in seconds/steps')
    parser.add_argument("-nogui", default=False, action='store_true', dest='nogui', help='disable gui, default: False')
    parser.add_argument("-scale", type=float, default=1.4, dest='scale', help='vehicle generation scale parameter, higher values generates more vehicles, default: 1.0')
    parser.add_argument("-demand", type=str, default='dynamic', dest='demand', help='vehicle demand generation patter, single limits vehicle network population to one, dynamic creates changing vehicle population, default:dynamic, options:single, dynamic')

    parser.add_argument("-offset", type=float, default=0.25, dest='offset', help='max sim offset fraction of total sim length, default: 0.3')

    #shared tsc params
    parser.add_argument("-gmin", type=int, default=5, dest='g_min', help='minimum green phase time (s), default: 5')
    parser.add_argument("-y", type=int, default=2, dest='y', help='yellow change phase time (s), default: 2')
    parser.add_argument("-r", type=int, default=3, dest='r', help='all red stop phase time (s), default: 3')

    #websters params
    parser.add_argument("-cmin", type=int, default=60, dest='c_min', help='minimum cycle time (s), default: 60')
    parser.add_argument("-cmax", type=int, default=180, dest='c_max', help='maximum cycle time (s), default: 180')
    parser.add_argument("-satflow", type=float, default=0.38, dest='sat_flow', help='lane vehicle saturation rate (veh/s), default: 0.38')
    parser.add_argument("-f", type=int, default=900, dest='update_freq', help='interval over which websters timing are computed (s), default: 900')

    #maxpressure params

    #self organizing traffic lights
    parser.add_argument("-theta", type=int, default=45, dest='theta', help='threshold to change signal (veh*s), default: 45')
    parser.add_argument("-omega", type=int, default=1, dest='omega', help='sotl param (veh*s), default: 1')
    parser.add_argument("-mu", type=int, default=3, dest='mu', help='sotl param(veh*s), default: 3')

    #rl params
    parser.add_argument("-eps", type=float, default=0.01, dest='eps', help='reinforcement learning explortation rate, default: 0.01')
    parser.add_argument("-nsteps", type=int, default=1, dest='nsteps', help='n step returns/max experience trajectory, default: 1')
    parser.add_argument("-nreplay", type=int, default=10000, dest='nreplay', help='maximum size of experience replay, default: 10000')
    parser.add_argument("-batch", type=int, default=32, dest='batch', help='batch size to sample from replay to train neural net, default: 32')
    parser.add_argument("-gamma", type=float, default=0.99, dest='gamma', help='reward discount factor, default: 0.99')
    parser.add_argument("-updates", type=int, default=10000, dest='updates', help='total number of batch updates for training, default: 10000')
    parser.add_argument("-target_freq", type=int, default=50, dest='target_freq', help='target network batch update frequency, default: 50')

    #neural net params
    parser.add_argument("-lr", type=float, default=0.0001, dest='lr', help='ddpg actor/dqn neural network learning rate, default: 0.0001')
    parser.add_argument("-lrc", type=float, default=0.001, dest='lrc', help='ddpg critic neural network learning rate, default: 0.001')
    parser.add_argument("-lre", type=float, default=0.00000001, dest='lre', help='neural network optimizer epsilon, default: 0.00000001')
    parser.add_argument("-hidden_act", type=str, default='elu', dest='hidden_act', help='neural network hidden layer activation, default: elu')
    parser.add_argument("-n_hidden", type=int, default=3, dest='n_hidden', help='neural network hidden layer scaling factor, default: 3')
    
    parser.add_argument("-save_path", type=str, default='saved_models', dest='save_path', help='dir to save neural network weights, default: saved_models')
    parser.add_argument("-save_replay", type=str, default='saved_replays', dest='save_replay', help='dir to save experience replays, default: saved_replays')
    parser.add_argument("-load_replay", default=False, action='store_true', dest='load_replay', help='load experience replays if they exist')

    parser.add_argument("-save_t", type=int, default=120, dest='save_t', help='interval in seconds between saving neural networks on learners, default: 120 (s)')
    parser.add_argument("-save", default=False, action='store_true', dest='save', help='use argument to save neural network weights')
    parser.add_argument("-load", default=False, action='store_true', dest='load', help='use argument to load neural network weights assuming they exist')

    #ddpg rl params
    parser.add_argument("-tau", type=float, default=0.005, dest='tau', help='ddpg online/target weight shifting tau, default: 0.005')
    parser.add_argument("-gmax", type=int, default=30, dest='g_max', help='maximum green phase time (s), default: 30')

    args = parser.parse_args()
    return args

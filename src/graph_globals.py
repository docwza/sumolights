import matplotlib
import matplotlib.pyplot as plt
import numpy as np

###https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

def global_params():
    #for line graphs
    scale = 10.0
    ###for pie graphs
    #scale = 5.0

    big = 5*scale
    med = 3.5*scale
    small = 2*scale
    
    plt.rcParams["font.family"] = "FreeSerif" 
    plt.rcParams["font.size"] = small 
    
    plt.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rc('axes', titlesize=med)
    matplotlib.rc('axes', labelsize=small)

    matplotlib.rc('xtick', labelsize=small)
    matplotlib.rc('ytick', labelsize=small)

    matplotlib.rc('legend', fontsize=small)
    matplotlib.rc('figure', titlesize=big)

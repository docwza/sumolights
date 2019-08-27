import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mpl

#from graph_globals import global_params

#legends code
#http://queirozf.com/entries/matplotlib-examples-displaying-and-configuring-legends
#good source for fancy graphs
#https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/

def graph(ax, data, graph_func, xtitle=None, ytitle_pad=None, title=None, legend=None, grid=None, xlim=None, ylim=None, colours=None):
    ####graph the function G
    #elements, data_labels = graph_func
    elements, labels = graph_func

    ###set the title and other such common elements
    if xtitle:
        ax.set_xlabel(xtitle)
                                                                   
    if ytitle_pad:
        ax.set_ylabel(ytitle_pad[0], rotation=360, labelpad=ytitle_pad[1])
                                                                   
    if title:
        ax.set_title(title)

    if legend:
        #ax.legend(elements, labels, loc=legend, framealpha=1.0)
        if colours is None:
            ax.legend(elements, labels, loc=legend, framealpha=1.0)
        else:
            patches = [ mpatches.Patch(color=colours[l], label=l) for l in labels]
            ax.legend(handles=patches, loc=legend, framealpha=1.0)

    if grid:
        ax.grid(linestyle='--')

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

def boxplot(ax, data, colors, data_labels):
    #bp = ax.boxplot( data, labels = data_labels, patch_artist=True )
    bp = ax.boxplot( data, notch=False, labels=['']*len(data_labels), patch_artist=True )

    for b, c in zip(bp['medians'], colors):
        ###white for all medians
        b.set(  color='w', linewidth=3)

    for b, c in zip(bp['boxes'], colors):
        ###color boxes specific color
        b.set( color = c )

    for b, c in zip(bp['fliers'], colors):
        ###color boxes specific color
        b.set( marker='+',  markeredgecolor=c, color = c, markersize=5 )


    for attr in ['caps', 'whiskers']:
        boxplots = np.array(bp[attr])
        ###these are doubles, because each boxplot has two of each
        boxplots = boxplots.reshape( (len(data_labels), 2) )
        
        for t,c in zip(boxplots, colors):
            for b in t:
                b.set( color=c, linewidth=2)
    return bp['boxes'], data_labels

def plot_mean_and_CI(ax, mean, lb, ub, color_mean, color_shading, label):
    ###https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/
    # plot the shaded range of the confidence intervals
    lb[ lb < 0 ] = 0.0
    ax.fill_between(range(mean.shape[0]), ub, lb, color=color_shading, alpha=.35, label=label)
    # plot the mean on top
    line, = ax.plot(mean, color_mean)
    return line

def multi_line_with_CI(ax, data, colors, data_labels):
    ###say we simulating different methods in some simulator, measuring some quantity at each step in sim
    ####data should be in form (method , sim run, measure)

    #fig.suptitle('Main figure title') main title for entire fig
    lines = []
    for d, c, l in zip(data, colors, data_labels):
        mean = np.mean(d, axis = 0)
        std = np.std(d, axis = 0)
        std_err = std / np.sqrt(d.shape[0])
        ###95% CI
        std_err *= 1.96
        line = plot_mean_and_CI( ax, mean, mean-std_err, mean+std_err, c, c, l)
        lines.append(line)
    return lines, data_labels

def multi_line(ax, data, colors, data_labels):
    ###say we simulating different methods in some simulator, measuring some quantity at each step in sim
    ####data should be in form (method , sim run, measure)

    #fig.suptitle('Main figure title') main title for entire fig
    lines = []
    for d, c, l in zip(data, colors, data_labels):
        line, = ax.plot(d, color=c)
        lines.append(line)
    return lines, data_labels

def scatter(ax, x, y, colors, data_labels, scatter_size=100):
    lines = []
    ax.scatter(x, y, color=colors, s=scatter_size, alpha=0.5)
    return lines, data_labels

def get_cmap(n, name='brg'):
    #taken from https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def multi_histogram(ax, data, colors, data_labels, dmin, dmax, nbins):
    n = len(data)
    bwidth = 1.0/n
    bars = []
    for d, c, i in zip(data, colors, range(n)):
        #get hist bins data
        hist, bin_edges = np.histogram(d, bins=nbins, range=[dmin, dmax])
        #offset bars for multiple histograms in same bin
        x = np.arange(len(hist))+(bwidth*i)
        #draw histo bars on graph
        bar = ax.bar(x, hist, width=bwidth, align='edge', color=c ) 
        bars.append(bar)

    #edit x-axis ticks to reflect bin edges
    bin_edges = bin_edges.astype(int)
    ax.set_xticks(np.arange(len(bin_edges)))
    #show every other one
    ax.set_xticklabels([ '' if i%2==1 else str(be) for be,i in zip(bin_edges,range(len(bin_edges)))  ])

    return bars, data_labels

def save_graph(f, fname, dpi, h, w):
    f = plt.gcf()
    f.set_size_inches(w, h, forward=True)
    f.savefig(fname, dpi=dpi, bbox_inches='tight')

'''
if __name__ == '__main__':
    global_params()
    colors = ['b','r','g','y']
    data_labels = ['Random', 'Actuated', 'Q-learn', 'A3C' ]
    f, axarr = plt.subplots(1,1)
    box_data = np.random.rand(100, 4)
    graph( axarr, box_data, boxplot( axarr, box_data, colors, data_labels), xtitle='XTITLE' , ytitle_pad = ('YTITLE',60) , title='MAIN TITLE', legend=(0.92, 0.92), grid=True)
    mean = [1.0, 2.0, 5.0, 2.8]
    std = [0.2, 0.7, 0.1, 2.4]
    data = [np.random.uniform(m,s,(30,300)) for m, s in zip(mean, std)]

    f, axarr = plt.subplots(1,1)
    graph( axarr, data, multi_line_with_CI(axarr, np.stack(data), colors, data_labels), xtitle='XTITLE' , ytitle_pad = ('YTITLE',60) , title='MAIN TITLE', legend=(0.92, 0.92), grid=True, xlim=[0.0,300.0], ylim=[0.0,4.0])

    f.suptitle('Main figure title')
    plt.show()    
'''

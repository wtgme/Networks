# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:40:27 2015

@author: tw5n14
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# m = 3
# N = 900

# G = nx.barabasi_albert_graph(N, m)

# degree_list=nx.degree(G).values()

# kmin=min(degree_list)
# kmax=max(degree_list)

# bins=[float(k-0.5) for k in range(kmin,kmax+2,1)]
# density, binedges = np.histogram(degree_list, bins=bins, density=True)
# bins = np.delete(bins, -1)

# logBins = np.logspace(np.log10(kmin), np.log10(kmax),num=20)
# logBinDensity, binedges = np.histogram(degree_list, bins=logBins, density=True)
# logBins = np.delete(logBins, -1)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xscale('log')
# ax.set_yscale('log')

# plt.plot(bins,density,'x',color='black')
# plt.plot(logBins,logBinDensity,'x',color='blue')
def drop_zeros(a_list):
    return [i for i in a_list if i>0]

def log_binning(counter_dict,bin_count=35):
    max_x = log10(max(counter_dict.keys()))
    max_y = log10(max(counter_dict.values()))
    max_base = max([max_x,max_y])

    min_x = log10(min(drop_zeros(counter_dict.keys())))

    bins = np.logspace(min_x,max_base,num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    bin_means_y = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.values())[0] / np.histogram(counter_dict.keys(),bins)[0])
    bin_means_x = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.keys())[0] / np.histogram(counter_dict.keys(),bins)[0])

    return bin_means_x,bin_means_y
    
    
ba_g = nx.barabasi_albert_graph(10000,2)
ba_c = nx.degree_centrality(ba_g)
# To convert normalized degrees to raw degrees
#ba_c = {k:int(v*(len(ba_g)-1)) for k,v in ba_c.iteritems()}
ba_c2 = dict(Counter(ba_c.values()))

ba_x,ba_y = log_binning(ba_c2,50)

plt.xscale('log')
plt.yscale('log')
plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)
plt.scatter(ba_c2.keys(),ba_c2.values(),c='b',marker='x')
plt.xlim((1e-4,1e-1))
plt.ylim((.9,1e4))
plt.xlabel('Connections (normalized)')
plt.ylabel('Frequency')
plt.show()
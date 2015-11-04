# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 16:22:26 2015

@author: tw5n14
"""

from networkx import *
import matplotlib.pyplot as plt
import numpy as np
import csv
import math


G = Graph()
userps = {} 
gw, cw = [],[] 

#load user profile from file
with open('poi.csv', 'rt') as f:
    reader = csv.reader(f)
    first_row = next(reader)
    for row in reader:
#        print row
        userp = {}
        userp['screem_name'] = row[1]
        userp['datetime'] = row[2]
        userp['descrip'] = row[3]
        userp['lan'] = row[6]
        userp['location'] = row[7]
        userp['gender'] = row[8]
        userp['gw'] = row[9]
        userp['cw'] = row[10]
        if row[9]!='' and row[10]!='':
#            print float(row[9]), float(row[10])
            gw.append(float(row[9]))
            cw.append(float(row[10]))
        userps[row[0]] = userp
        del userp 

# load a network from file
with open('mrredges-no-tweet-no-retweet-poi-counted.txt', 'r') as fo:
    for line in fo.readlines():
        tokens = line.split(',')
        n1 = (tokens[0])
        n2 = (tokens[1])
        b_type = tokens[2]
        weightv = int(tokens[3])
        # reply-to mentioned
#        if b_type == 'reply-to':
        if (G.has_node(n1)) and (G.has_node(n2)) and (G.has_edge(n1, n2)):
#            print n1, n2, G.has_node(n1), G.has_node(n2), G.has_edge(n1,n2), weightv, G[n2]
            G[n1][n2]['weight'] += weightv
        else:
            G.add_edge(n1, n2, weight=weightv)
            
    
        
# pos = random_layout(G)
# pos = shell_layout(G)
# pos = spring_layout(G)
#pos = spectral_layout(G)
# draw(G, pos)
# plt.show()

def pearson(x,y):
    n = len(x)
    avg_x = float(sum(x))/n
    avg_y = float(sum(y))/n
    print avg_x, avg_y
    diffprod = 0.0
    xdiff2 = 0.0
    ydiff2 = 0.0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff*ydiff
        xdiff2 += xdiff*xdiff
        ydiff2 += ydiff*ydiff
    return diffprod/math.sqrt(xdiff2*ydiff2)

def drop_zeros(list_a):
    return [i for i in list_a if i>0]

def log_binning(list_x, list_y, bin_count=35):
    max_x = np.log10(max(list_x))
    max_y = np.log10(max(list_y))
    min_x = np.log10(min(drop_zeros(list_x)))
    min_y = np.log10(min(drop_zeros(list_y)))
    bins_x = np.logspace(min_x, max_x, num=bin_count)
    bins_y = np.logspace(min_y, max_y, num=bin_count)
    bin_means_x = (np.histogram(list_x, bins_x, weights=list_x))[0] / (np.histogram(list_x, bins_x)[0])
    bin_means_y = (np.histogram(list_y, bins_y, weights=list_y))[0] / (np.histogram(list_y, bins_y)[0])    
    return bin_means_x, bin_means_y


def PD(list_x, bin_count=35):  
    max_x = np.log10(max(list_x))
    min_x = np.log10(min(drop_zeros(list_x)))
    bins_x = np.logspace(min_x, max_x, num=bin_count)
    weights = np.ones_like(list_x)/float(len(list_x))
    hist, bin_deges = np.histogram(list_x, bins_x, weights=weights)
    return hist, bin_deges

def CPD(list_x, bin_count=35):  
    max_x = np.log10(max(list_x))
    min_x = np.log10(min(drop_zeros(list_x)))
    bins_x = np.logspace(min_x, max_x, num=bin_count)
    weights = np.ones_like(list_x)/float(len(list_x))
    hist, bin_deges = np.histogram(list_x, bins_x, weights=weights)
    cum = np.cumsum(hist[::-1])[::-1] 
#    print len(cum)
#    print len(bin_deges)
    return cum, bin_deges
    

#network analysis
print 'The number of nodes: %d' %(G.order())
print 'The number of nodes: %d' %(G.__len__())
print 'The number of nodes: %d' %(G.number_of_nodes())
print 'The number of edges: %d' %(G.size())
print 'The number of self-loop: %d' %(G.number_of_selfloops())


#gwhist, gwbins = CPD(gw,50)
#gwbin_centers = (gwbins[1:]+gwbins[:-1])/2.0
#print gwhist, sum(gwhist)
#gwp, = plt.plot(gwbin_centers, gwhist, color='blue', marker='x')
#cwhist, cwbins = CPD(cw,50)
#cwbin_centers = (cwbins[1:]+cwbins[:-1])/2.0
#cwp, = plt.plot(cwbin_centers, cwhist, color='red', marker='.')
#print cwhist, sum(cwhist)
#plt.legend((gwp, cwp), ('Global-Weight','Current-Weight'), loc=4)
#plt.xscale('log')
#plt.yscale('linear')
#plt.xlabel('KG(x)')
#plt.ylabel('P(x)')

print 'The plot of in-degree and out-degree of nodes'
print 'Node \t degree \t Strength'
degree, strength = [],[]
for node in G.nodes():
    print '%s \t %d \t %d' %(node, G.degree(node), G.degree(node, weight='weight'))
    degree.append(G.degree(node))
    strength.append(G.degree(node, weight='weight'))

deg, stre = log_binning(degree, strength, 50)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Strength')
plt.xlim(1, 1e4)
plt.ylim(1, 1e4)
degp = plt.scatter(deg, stre, c='r', marker='s', s=50, alpha=0.5, label='Undirected Graph(p=0.76)')
plt.legend(handles=[degp])
#plt.legend((degp), ('Undirected Graph(p=0.76)'), loc=4)
print 'pearson correlation of instrength and outstrength: %f' %(pearson(degree, strength))

    

#indcum, indbin_deges = CPD(degree, 100)
#indbin_centers = (indbin_deges[1:]+indbin_deges[:-1])/2.0
#indegr, = plt.plot(indbin_centers, indcum, color='blue', marker='x')
#
#inscum, insbin_deges = CPD(strength, 100)
#insbin_centers = (insbin_deges[1:]+insbin_deges[:-1])/2.0
#instre, = plt.plot(insbin_centers, inscum, color='red', marker='.')
#
#plt.legend((indegr, instre), ('Degree','Strength'), loc=3)
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel('k')
#plt.ylabel('P(k)')
    
    

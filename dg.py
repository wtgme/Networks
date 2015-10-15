# -*- coding: utf-8 -*-
from networkx import *
import math
import matplotlib.pyplot as plt
import numpy as np

DG = DiGraph()
UserDic = {}

class User(object):
    """docstring for User"""
    def __init__(self, uid, screen_name, datetime, descrip, location, gender):
        super(User, self).__init__()
        self.uid = uid
        self.screen_name = screen_name
        self.datetime = datetime
        self.descrip = descrip
        self.location = location
        self.gender = gender
    

# load a network from file
with open('mrredges-no-tweet-no-retweet-poi-counted.txt', 'r') as fo:
    for line in fo.readlines():
        tokens = line.split(',')
        n1 = (tokens[0])
        n2 = (tokens[1])
        b_type = tokens[2]
        weightv = int(tokens[3])
        # reply-to mentioned
        if (DG.has_node(n1)) and (DG.has_node(n2)) and (DG.has_edge(n1, n2)):
            DG[n1][n2]['weight'] += weightv
        else:
            DG.add_edge(n1, n2, weight=weightv)

        
# pos = random_layout(DG)
# pos = shell_layout(DG)
#pos = spring_layout(DG)
#pos = spectral_layout(DG)
#draw(DG, pos)
#plt.show()
#plt.title('Plot of Network(reply)')
#pos = spring_layout(DG)
## pos = spectral_layout(DG)
#draw(DG, pos)
#plt.show()


# load user profile from file
# with open('poi.txt','r') as fo:
#     for line in fo.readlines():
#         tokens = line.split(',')
#         user = User(tokens[0], tokens[1], tokens[2], tokens[3], tokens[7], tokens[8])
#         UserDic[user.uid] = user

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


def CPD(list_x, bin_count=35):  
    max_x = np.log10(max(list_x))
    min_x = np.log10(min(drop_zeros(list_x)))
    bins_x = np.logspace(min_x, max_x, num=bin_count)
    weights = np.ones_like(list_x)/float(len(list_x))
    hist, bin_deges = np.histogram(list_x, bins_x, weights=weights)
#    cum = np.cumsum(hist)
    cum = np.cumsum(hist[::-1])[::-1] 
    print len(cum)
    print len(bin_deges)
    return cum, bin_deges

    

##network analysis
print 'The number of nodes: %d' %(DG.order())
print 'The number of nodes: %d' %(DG.__len__())
print 'The number of nodes: %d' %(DG.number_of_nodes())
print 'The number of edges: %d' %(DG.size())
print 'The number of self-loop: %d' %(DG.number_of_selfloops())

print 'The plot of in-degree and out-degree of nodes'
print 'Node \t In-degree \t Out-degree'
indegree, outdegree, instrength, outstrength = [],[],[],[]
for node in DG.nodes():
#    print 'Degree: %s \t %d \t %d \t %d' %(node, DG.in_degree(node), DG.out_degree(node), DG.degree(node))
#    print 'Strength: %s \t %d \t %d \t %d' %(node, DG.in_degree(node, weight='weight'), DG.out_degree(node, weight='weight'), DG.degree(node, weight='weight'))   
    indegree.append(DG.in_degree(node))
    outdegree.append(DG.out_degree(node))
    instrength.append(DG.out_degree(node, weight='weight'))
    outstrength.append(DG.out_degree(node, weight='weight'))

bd_in, bd_out = log_binning(indegree, outdegree, 50)
bs_in, bs_out = log_binning(instrength, outstrength, 50)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('In-count')
plt.ylabel('Out-count')
plt.xlim(1, 1e3+1000)
plt.ylim(1, 1e3+1000)
degree = plt.scatter(bd_in, bd_out, c='r', marker='s', s=50, alpha=0.5)
strength = plt.scatter(bs_in, bs_out, c='b', marker='o', s=50, alpha=0.5)
plt.legend((degree, strength), ('Degree(p=0.62)', 'Strength(p=1.00)'), loc='upper left')


#indcum, indbin_deges = CPD(indegree, 100)
#indbin_centers = (indbin_deges[1:]+indbin_deges[:-1])/2.0
#indegr, = plt.plot(indbin_centers, indcum, color='blue', marker='*')
#
#outdcum, outdbin_deges = CPD(outdegree, 100)
#outdbin_centers = (outdbin_deges[1:]+outdbin_deges[:-1])/2.0
#outdegr, = plt.plot(outdbin_centers, outdcum, color='black', marker='+')
#
#inscum, insbin_deges = CPD(instrength, 100)
#insbin_centers = (insbin_deges[1:]+insbin_deges[:-1])/2.0
#instre, = plt.plot(insbin_centers, inscum, color='red', marker='x')
#
#outscum, outsbin_deges = CPD(outstrength, 100)
#outsbin_centers = (outsbin_deges[1:]+outsbin_deges[:-1])/2.0
#outstre, = plt.plot(outsbin_centers, outscum, color='c', marker='.', linewidth=3)
#
#plt.legend((indegr, outdegr, instre, outstre), ('In-Degree', 'Out-Degree', 'In-Strength', 'Out-Strength'), loc='right')
#
#plt.xscale('log')
#plt.yscale('log')

print 'pearson correlation of indegree and outdegree: %f' %(pearson(indegree, outdegree))
print 'pearson correlation of instrength and outstrength: %f' %(pearson(instrength, outstrength))





    

##plot the numbers of nodes with degree K
#plt.title('Numbers of Nodes with In-degree K Plot(mention)')
#plt.ylabel('Number of nodes')
#plt.xlabel('Degree')
#degseq=list(DG.in_degree(weight='weight').values())
#print degseq
#dmax=max(degseq)+1
#freq= [ 0 for d in range(dmax) ]
#for d in degseq:
#    freq[d] += 1
#plt.plot(freq)
##plt.plot(list(utils.cumulative_sum(freq)))
#plt.show()


#def cumu_dist(degree_list):
#    kmin=min(degree_list)
#    kmax=max(degree_list)
#    
#    bins=[float(k-0.5) for k in range(kmin,kmax+2,1)]
#    density, binedges = np.histogram(degree_list, bins=bins, density=True)
#    bins = np.delete(bins, -1)
#    
#    logBins = np.logspace(np.log10(kmin), np.log10(kmax),num=50)
#    logBinDensity, binedges = np.histogram(degree_list, bins=logBins, density=True)
#    logBins = np.delete(logBins, -1)
##    fig = plt.figure()
##    ax = fig.add_subplot(111)
##    ax.set_xscale('log')
##    ax.set_yscale('log')
##    plt.plot(bins,density,'+',color='black')
##    plt.plot(logBins,logBinDensity,'x',color='blue')
#    return logBins, logBinDensity
    


#plot cumulative distribution of degree K
#plt.title('Cumulative Distribution of Nodes with Degree K Plot(mention)')
#plt.ylabel('P')
#plt.ylim(0.0,1.1)
#plt.xlabel('Degree')
#degseq=list(DG.degree(weight='weight').values())
#logBins, logBinDensity = cumu_dist(degseq)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_title('Cumulative Distribution of Nodes with Degree K Plot(mention)')
#ax.set_xscale('log')
#ax.set_yscale('log')
#plt.plot(logBins,logBinDensity,'x',color='blue')
#dmax=max(degseq)+1
#freq= [ 0 for d in range(dmax) ]
#for d in degseq:
#    freq[d] += 1
##plt.plot(freq)
#sumall = sum(freq)
#cumul = []
#temp = 0.0
#for fre in freq:
#    temp += fre
#    cumul.append(temp/sumall)

#plt.plot(cumul)
#plt.show()


#histogram of path lengths
#print 'source vertex {taget: length,}'
#pathlengths = []
#for v in DG.nodes():
#    spl = single_source_shortest_path_length(DG, v)
##    print '%s %s' %(v, spl)
#    for p in spl.values():
#        pathlengths.append(p)
#print 'average shortest path length %s' %(float(sum(pathlengths))/len(pathlengths))
#
#dist = {}
#for p in pathlengths:
#    v = dist.get(p,0)+1
#    dist[p] = v
#print 'length #paths'
#verts = dist.keys()
#distlist = []
#for d in sorted(verts):
#    distlist.append(dist[d])
#    print '%s %d' %(d, dist[d])
#plt.title('Plot of Shortest Path and Numbers of Paths(reply)')
#plt.ylabel('Counts')
#plt.xlabel('Path Length')
##plt.ylim(ymin=-10.0)
#plt.plot(distlist)

#print (is_connected(DG))

#print 'radius: %d' %(radius(DG))
#print 'diameter: %d' %(diameter(DG))
#print 'eccentricity: %s' %(eccentricity(DG))
#print 'center: %s' %(center(DG))
#print 'periphery: %s' %(periphery(DG))
#print 'density: %s' %(density(DG))

#k_clique_communities(DG, 5)
#draw(DG)
#plt.show()
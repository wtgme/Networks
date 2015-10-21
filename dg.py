# -*- coding: utf-8 -*-

from networkx import *
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.stats
import powerlaw


DG = DiGraph()
UserDic = {}
userps = {}  

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
#        print userp
        userps[row[0]] = userp
        del userp 

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
#    the returned values are raw values, not logarithmic values
    max_x = np.log10(max(list_x))
    max_y = np.log10(max(list_y))
    min_x = np.log10(min(drop_zeros(list_x)))
    min_y = np.log10(min(drop_zeros(list_y)))
    bins_x = np.logspace(min_x, max_x, num=bin_count)
    bins_y = np.logspace(min_y, max_y, num=bin_count)
    bin_means_x = (np.histogram(list_x, bins_x, weights=list_x))[0].astype(float) / (np.histogram(list_x, bins_x)[0])
    bin_means_y = (np.histogram(list_y, bins_y, weights=list_y))[0].astype(float) / (np.histogram(list_y, bins_y)[0])    
    return bin_means_x, bin_means_y


def CPD(list_x, bin_count=35):  
    max_x = np.log10(max(list_x))
    min_x = np.log10(min(drop_zeros(list_x)))
    bins_x = np.logspace(min_x, max_x, num=bin_count)
    weights = np.ones_like(list_x)/float(len(list_x))
    hist, bin_deges = np.histogram(list_x, bins_x, weights=weights)
#    cum = np.cumsum(hist)
    cum = np.cumsum(hist[::-1])[::-1] 
#    print len(cum)
#    print len(bin_deges)
    return cum, bin_deges

def log_fit(list_x, list_y):
    X = np.asarray(list_x, dtype=float)
    Y = np.asarray(list_y, dtype=float)
    logX = np.log10(X)
    logY = np.log10(Y)
    coefficients = np.polyfit(logX, logY, 1)
    polynomial = np.poly1d(coefficients)
    Y_fit = polynomial(X)
    return Y_fit

##network analysis
print 'The number of nodes: %d' %(DG.order())
print 'The number of nodes: %d' %(DG.__len__())
print 'The number of nodes: %d' %(DG.number_of_nodes())
print 'The number of edges: %d' %(DG.size())
print 'The number of self-loop: %d' %(DG.number_of_selfloops())

print 'The plot of in-degree and out-degree of nodes'
print 'Node \t In \t Out \t In+Out'
indegree, outdegree, instrength, outstrength = [],[],[],[]
for node in DG.nodes():
#    print 'Degree: %s \t %d \t %d \t %d' %(node, DG.in_degree(node), DG.out_degree(node), DG.degree(node))
#    print 'Strength: %s \t %d \t %d \t %d' %(node, DG.in_degree(node, weight='weight'), DG.out_degree(node, weight='weight'), DG.degree(node, weight='weight'))   
    indegree.append(DG.in_degree(node)+1)
    outdegree.append(DG.out_degree(node)+1)
    instrength.append(DG.in_degree(node, weight='weight')+1)
    outstrength.append(DG.out_degree(node, weight='weight')+1)

#indegree = drop_zeros(indegree)
#outdegree = drop_zeros(outdegree)
#instrength = drop_zeros(instrength)
#outstrength = drop_zeros(outstrength)

#instrength.extend(outstrength)

fit = powerlaw.Fit(indegree, discrete=True)
figPDF = fit.plot_pdf(color='b', linewidth=2, label=r"Empirical, no $x_{max}$")
fit.power_law.plot_pdf(color='b', linestyle='--', ax=figPDF, label=r"Empirical, no $x_{max}$")
print 'alpha:', fit.power_law.alpha
print 'error:', fit.power_law.sigma

fit2 = powerlaw.Fit(outdegree, discrete=True)
fit2.plot_pdf(color='r', linewidth=2, ax=figPDF, label=r"Empirical, no $x_{max}$")
fit2.power_law.plot_pdf(color='r', linestyle='--', ax=figPDF, label=r"Empirical, no $x_{max}$")
print 'alpha:', fit2.power_law.alpha
print 'error:', fit2.power_law.sigma

figPDF.set_ylabel("P(X)")
figPDF.set_xlabel("In-degree")
plt.savefig('degreepdf.eps', bbox_inches='tight')


#bd_in, bd_out = log_binning((indegree), (outdegree), 15)
#bs_in, bs_out = log_binning((outstrength), (instrength), 15)

#print bd_in
#print bd_out


#plt.plot(logB, logA)
#plt.plot(bs_in, log_fit(bs_in,bs_out), 'b--')
#plt.plot(bd_out, log_fit(bd_out,bd_in), 'r--')


#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(bs_in,bs_out)
#print slope, intercept
#plt.plot(bs_in, [slope*b_in+intercept for b_in in bs_in], '--r')
#plt.plot(bs_in, np.power(10, intercept) * np.power(bs_in, slope), '--r')
#ax.plot(X, [m*x + c for x in X], 'r')

#print fit_fn(bd_in)
#
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel('In')
#plt.ylabel('Out')
#plt.xlim(1, 1e4)
#plt.ylim(1, 1e4)
#degree = plt.scatter(bd_in, bd_out, c='r', marker='s', s=50, alpha=0.5)
#strength = plt.scatter(bs_in, bs_out, c='b', marker='o', s=50, alpha=0.5)
#plt.legend((degree, strength), ('In(cov(in,out) = 0.777)', 'Out(cov(in,out) = 0.721)'), loc='upper left')


print 'pearson correlation of indegree and outdegree: %f' %(pearson(indegree, instrength))
print 'pearson correlation of instrength and outstrength: %f' %(pearson(outdegree, outstrength))


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
#plt.legend((indegr, outdegr, instre, outstre), ('In-Degree', 'Out-Degree', 'In-Strength', 'Out-Strength'), loc=3)
#
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel('k')
#plt.ylabel('P(k)')

#print instrength[1], outstrength[1]
##print zip(instrength, outstrength)
#print 'pearson correlation of indegree and outdegree: %f' %(pearsonr(instrength, outstrength)[0])
#print pearsonr(indegree, outdegree)[1]


#print 'pearson correlation of indegree and outdegree: %f' %(pearsonr(indegree, outdegree)[0])
#print 'pearson correlation of instrength and outstrength: %f' %(pearsonr(instrength, outstrength)[0])


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
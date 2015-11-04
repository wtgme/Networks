# -*- coding: utf-8 -*-

from networkx import *
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
import powerlaw
from sklearn.metrics import mean_squared_error
import os


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

#with open('mrredges-no-tweet-no-retweet-poi-counted.csv', 'rt') as f:
#    reader = csv.reader(f)
#    first_row = next(reader)
#    for row in reader:
#        u1 = row[0]
#        u2 = row[1]
#        btype = row[2]
#        count = row[3]
#        userID = userIdMap.get(u1,len(userIdMap))
#        userIdMap[u1] = userID
#        userID = userIdMap.get(u2,len(userIdMap))
#        userIdMap[u2] = userID
#        print u1, u2, btype, count

file_path = os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1])+'/data/mrredges-no-tweet-no-retweet-poi-counted.txt'
print os.path.dirname(__file__).split(os.sep)[:-1]
print os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
with open(file_path, 'r') as fo:
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

A = adjacency_matrix(DG, weight=None)
#A = adjacency_matrix(DG)
# print A
Ade = A.todense()

Nlist = map(int, DG.nodes())
print len(Nlist)

poi = {}
file_path = os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1])+'/data/poi.csv'

f = open(file_path, 'rb')
reader = csv.reader(f, lineterminator='\n')
first_row = next(reader)
for row in reader:
    des = row[3]
    row[3] = des.replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ').replace('\n\r', ' ')
    # print '-------------'
    # print row[3]
    poi[row[0]] = row

# print 'Output poi'
# csvfile = open('targeted-poi.csv', 'wb')
# spamwriter = csv.writer(csvfile)
# spamwriter.writerow(first_row)
# for index in xrange(len(Nlist)):
#     # print poi.get(str(Nlist[index]))
#     spamwriter.writerow(poi.get(str(Nlist[index]), None))



# with open('degree_adjacency_matrix.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow([0]+Nlist)
#     for index in xrange(len(Nlist)):
#         spamwriter.writerow([Nlist[index]] + Ade[index].getA1().tolist())


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



def pearson(x,y):
    n = len(x)
    avg_x = float(sum(x))/n
    avg_y = float(sum(y))/n
    print 'The means of two lists:', avg_x, avg_y
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

def RMSE(predict, truth):
    RMSE = mean_squared_error(truth, predict)**0.5
    return RMSE

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

def power_law_fit(list_x, label_x, savename='figure', list_y = None, Label_y = ''):
    plt.clf()
    fit = powerlaw.Fit(list_x, discrete=True)
    figPDF = fit.plot_pdf(color='b', linewidth=2, label=r"Empirical, "+label_x)
    fit.power_law.plot_pdf(color='b', linestyle='--', ax=figPDF, label=r"Fit, "+label_x)
    print 'alpha:', fit.power_law.alpha
    print 'error:', fit.power_law.sigma
    
    if list_y != None:
        fit = powerlaw.Fit(list_y, discrete=True)
        fit.plot_pdf(color='r', linewidth=2, ax=figPDF, label=r"Empirical, "+Label_y)
        fit.power_law.plot_pdf(color='r', linestyle='--', ax=figPDF, label=r"Fit, "+Label_y)
        print 'alpha:', fit.power_law.alpha
        print 'error:', fit.power_law.sigma
    
    figPDF.set_ylabel("p(k)")
    figPDF.set_xlabel("k")
    handles, labels = figPDF.get_legend_handles_labels()
    leg = figPDF.legend(handles, labels, loc=3)
    leg.draw_frame(False)
    plt.savefig(savename+'.eps', bbox_inches='tight')

def log_fit(list_x, list_y):
    X = np.asarray(list_x, dtype=float)
    Y = np.asarray(list_y, dtype=float)
    logX = np.log10(X)
    logY = np.log10(Y)
    coefficients = np.polyfit(logX, logY, 1)
    polynomial = np.poly1d(coefficients)
    print 'Polynomial:', polynomial
    logY_fit = polynomial(logX)
    print 'Fitting RMSE(log)', RMSE(logY, logY_fit)
    print 'Fitting RMSE(raw)', RMSE(Y, np.power(10, logY_fit))
#    print Y
    return np.power(10, logY_fit)
#    return logX, logY_fit
    
def plot_log_fit(list_x, list_y, ax=None, **kwargs):
    if not ax:
        plt.plot(list_x, log_fit(list_x,list_y), **kwargs)
        ax = plt.gca()
    else:
        ax.plot(list_x, log_fit(list_x,list_y), **kwargs)  
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax  


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
    in_d, out_d, in_s, out_s = DG.in_degree(node), DG.out_degree(node), DG.in_degree(node, weight='weight'), DG.out_degree(node, weight='weight')
    indegree.append(in_d)
    outdegree.append(out_d)
    instrength.append(in_s)
    outstrength.append(out_s)

#indegree = drop_zeros(indegree)
#outdegree = drop_zeros(outdegree)
#instrength = drop_zeros(instrength)
#outstrength = drop_zeros(outstrength)

#instrength.extend(outstrength)

'''Power-law Fitting'''
#power_law_fit(indegree, 'in-degree', 'degreepdf1')
#power_law_fit(indegree, 'in-degree', 'degreepdf1',outdegree,'out-degree')


'''Log-Log fit degree and strength'''
#plt.clf()
#list_x_bined, list_y_bined = log_binning(instrength, outstrength, 12)
#plt.plot(list_x_bined, list_y_bined, 'bo', label='Empirical, $s_o(s_i)$')
#ax = plt.gca()
#plot_log_fit(list_x_bined, list_y_bined, ax=ax, color='b', linestyle='--',label='Fit, $s_o(s_i)$')
#
#list_x_bined, list_y_bined = log_binning(outstrength, instrength, 12)
#ax.plot(list_x_bined, list_y_bined, 'ro',label='Empirical, $s_i(s_o)$')
#plot_log_fit(list_x_bined, list_y_bined, ax=ax, color='r', linestyle='--', label='Fit, $s_i(s_o)$')
#ax.set_ylabel("s")
#ax.set_xlabel("s")
##handles, labels = ax.get_legend_handles_labels()
#ax.legend(loc=2)
##leg.draw_frame(False)
#plt.savefig('ss.eps', bbox_inches='tight')


#plot_log_fit(indegree, instrength, 'in-strength', 'out-strength', 15, 'strenghtlogfit')


#print 'pearson correlation of indegree and outdegree: %f' %(pearson(indegree, instrength))
#print 'pearson correlation of instrength and outstrength: %f' %(pearson(outdegree, outstrength))


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
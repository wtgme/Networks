# -*- coding: utf-8 -*-
"""
Created on 5:56 PM, 11/4/15

@author: wt

"""
from networkx import *
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
import mypowerlaw as powerlaw
from sklearn.metrics import mean_squared_error
import os


# load a network from file (directed weighted network)
def load_network():
    DG = DiGraph()
    file_path = os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1])+'/data/mrredges-no-tweet-no-retweet-poi-counted.csv'
    with open(file_path, 'rt') as fo:
        reader = csv.reader(fo)
        first_row = next(reader)
        for row in reader:
            n1 = (row[0])
            n2 = (row[1])
            b_type = row[2]
            weightv = int(row[3])
            # reply-to mentioned
            if (DG.has_node(n1)) and (DG.has_node(n2)) and (DG.has_edge(n1, n2)):
                DG[n1][n2]['weight'] += weightv
            else:
                DG.add_edge(n1, n2, weight=weightv)
    return DG


def get_adjacency_matrix(DG):
    A = adjacency_matrix(DG, weight=None)  # degree matrix, i.e. 1 or 0
    #A = adjacency_matrix(DG)  # strength matrix, i.e., the number of edge
    return A


def out_adjacency_matrix(DG):
    A = adjacency_matrix(DG, weight=None)  # degree matrix, i.e. 1 or 0
    #A = adjacency_matrix(DG)  # strength matrix, i.e., the number of edge
    Ade = A.todense()
    Nlist = map(int, DG.nodes())
    print len(Nlist)
    with open('degree_adjacency_matrix.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([0]+Nlist)
        for index in xrange(len(Nlist)):
            spamwriter.writerow([Nlist[index]] + Ade[index].getA1().tolist())


def load_poi():
    # Get profiles of all users
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
    # return the description in the FIRST row and contents
    return (first_row, poi)


def out_targted_poi(DG):
    # print 'Output poi in DG network'
    Nlist = map(int, DG.nodes())
    print len(Nlist)
    first_row, poi = load_poi()
    csvfile = open('targeted-poi.csv', 'wb')
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(first_row)
    for index in xrange(len(Nlist)):
        # print poi.get(str(Nlist[index]))
        spamwriter.writerow(poi.get(str(Nlist[index]), None))


def plot_whole_network(DG):
    # pos = random_layout(DG)
    # pos = shell_layout(DG)
    pos = spring_layout(DG)
    # pos = spectral_layout(DG)
    draw(DG, pos)
    plt.show()
    plt.title('Plot of Network(reply)')
    draw(DG, pos)
    plt.show()


def pearson(x, y):
    # calculate the pearson correlation of two list
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
    # discard the zeros in a list
    return [i for i in list_a if i>0]


def rmse(predict, truth):
    # calculate RMSE of a prediction
    RMSE = mean_squared_error(truth, predict)**0.5
    return RMSE


def CPD(list_x, bin_count=35):
    max_x = np.log10(max(list_x))
    min_x = np.log10(min(drop_zeros(list_x)))
    bins_x = np.logspace(min_x, max_x, num=bin_count)
    weights = np.ones_like(list_x)/float(len(list_x))
    hist, bin_deges = np.histogram(list_x, bins_x, weights=weights)
    # cum = np.cumsum(hist)
    cum = np.cumsum(hist[::-1])[::-1]
    # print len(cum)
    # print len(bin_deges)
    return cum, bin_deges


def log_binning(list_x, list_y, bin_count=35):
    # the returned values are raw values, not logarithmic values
    max_x = np.log10(max(list_x))
    max_y = np.log10(max(list_y))
    min_x = np.log10(min(drop_zeros(list_x)))
    min_y = np.log10(min(drop_zeros(list_y)))
    bins_x = np.logspace(min_x, max_x, num=bin_count)
    bins_y = np.logspace(min_y, max_y, num=bin_count)
    bin_means_x = (np.histogram(list_x, bins_x, weights=list_x))[0].astype(float) / (np.histogram(list_x, bins_x)[0])
    bin_means_y = (np.histogram(list_y, bins_y, weights=list_y))[0].astype(float) / (np.histogram(list_y, bins_y)[0])
    return bin_means_x, bin_means_y


def log_fit(list_x, list_y):
    X = np.asarray(list_x, dtype=float)
    Y = np.asarray(list_y, dtype=float)
    logX = np.log10(X)
    logY = np.log10(Y)
    coefficients = np.polyfit(logX, logY, 1)
    polynomial = np.poly1d(coefficients)
    print 'Polynomial:', polynomial
    logY_fit = polynomial(logX)
    print 'Fitting RMSE(log)', rmse(logY, logY_fit)
    print 'Fitting RMSE(raw)', rmse(Y, np.power(10, logY_fit))
    # print Y
    return np.power(10, logY_fit)
    # return logX, logY_fit


def plot_log_fit(list_x, list_y, ax=None, **kwargs):
    if not ax:
        plt.plot(list_x, log_fit(list_x,list_y), **kwargs)
        ax = plt.gca()
    else:
        ax.plot(list_x, log_fit(list_x,list_y), **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax


def log_pmd(data, number_bin):
    data = drop_zeros(data)
    minx = np.log10(min(data))
    maxx = np.log10(max(data))
    # example meaning: np.logspace(2.0, 3.0, num=4)--->array([  100.  ,   215.443469  ,   464.15888336,  1000.        ])
    binsx = np.logspace(minx, maxx, num=number_bin)
    counts = np.histogram(data, binsx)[0]
    means = (binsx[1:]+binsx[:-1])/2.0
    p = [float(c)/sum(counts) for c in counts]
    new_p, new_means = [], []
    for index in xrange(len(p)):
        if p[index] != 0.0:
            new_p.append(p[index])
            new_means.append(means[index])
    return (new_means, new_p)


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
    plt.show()
    # plt.savefig(savename+'.eps', bbox_inches='tight')


# network analysis
DG = load_network()
print 'The number of nodes: %d' %(DG.order())
print 'The number of nodes: %d' %(DG.__len__())
print 'The number of nodes: %d' %(DG.number_of_nodes())
print 'The number of edges: %d' %(DG.size())
print 'The number of self-loop: %d' %(DG.number_of_selfloops())

print 'The plot of in-degree and out-degree of nodes'
print 'Node \t In \t Out \t In+Out'
indegree, outdegree, instrength, outstrength = [],[],[],[]
for node in DG.nodes():
    # print 'Degree: %s \t %d \t %d \t %d' %(node, DG.in_degree(node), DG.out_degree(node), DG.degree(node))
    # print 'Strength: %s \t %d \t %d \t %d' %(node, DG.in_degree(node, weight='weight'), DG.out_degree(node, weight='weight'), DG.degree(node, weight='weight'))
    in_d, out_d, in_s, out_s = DG.in_degree(node), DG.out_degree(node), DG.in_degree(node, weight='weight'), DG.out_degree(node, weight='weight')
    indegree.append(in_d)
    outdegree.append(out_d)
    instrength.append(in_s)
    outstrength.append(out_s)


log_pmd(indegree, 30)
log_pmd(instrength, 30)


# indegree = drop_zeros(indegree)
# outdegree = drop_zeros(outdegree)
# instrength = drop_zeros(instrength)
# outstrength = drop_zeros(outstrength)

# instrength.extend(outstrength)

'''Power-law Fitting'''
# power_law_fit(indegree, 'in-degree', 'degreepdf1')
# power_law_fit(indegree, 'in-degree', 'degreepdf1',outdegree,'out-degree')


plt.clf()
rangx, proy = log_pmd(indegree, 30)
plt.plot(rangx, proy, 'bo', label='Empirical, indegree')
ax = plt.gca()
plot_log_fit(rangx, proy, ax=ax, color='b', linestyle='--',label='Fit, indegree')
rangx, proy = log_pmd(outdegree, 30)
ax.plot(rangx, proy, 'ro', label='Empirical, outdegree')
plot_log_fit(rangx, proy, ax=ax, color='r', linestyle='--',label='Fit, outdegree')


'''Log-Log fit degree and strength'''
# plt.clf()
# list_x_bined, list_y_bined = log_binning(instrength, outstrength, 12)
# plt.plot(list_x_bined, list_y_bined, 'bo', label='Empirical, $s_o(s_i)$')
# ax = plt.gca()
# plot_log_fit(list_x_bined, list_y_bined, ax=ax, color='b', linestyle='--',label='Fit, $s_o(s_i)$')
#
# list_x_bined, list_y_bined = log_binning(outstrength, instrength, 12)
# ax.plot(list_x_bined, list_y_bined, 'ro',label='Empirical, $s_i(s_o)$')
# plot_log_fit(list_x_bined, list_y_bined, ax=ax, color='r', linestyle='--', label='Fit, $s_i(s_o)$')
# ax.set_ylabel("s")
# ax.set_xlabel("s")
# # handles, labels = ax.get_legend_handles_labels()
# ax.legend(loc=2)
# # leg.draw_frame(False)
# plt.savefig('ss.eps', bbox_inches='tight')
plt.show()


# plot_log_fit(indegree, instrength, 'in-strength', 'out-strength', 15, 'strenghtlogfit')


# print 'pearson correlation of indegree and outdegree: %f' %(pearson(indegree, instrength))
# print 'pearson correlation of instrength and outstrength: %f' %(pearson(outdegree, outstrength))
#
# print 'radius: %d' %(radius(DG))
# print 'diameter: %d' %(diameter(DG))
# print 'eccentricity: %s' %(eccentricity(DG))
# print 'center: %s' %(center(DG))
# print 'periphery: %s' %(periphery(DG))
# print 'density: %s' %(density(DG))

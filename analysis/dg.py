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
from sklearn.metrics import mean_squared_error
import os
from collections import Counter
import powerlaw
from scipy.optimize import minimize
import scipy.stats as stats


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

def pdf(data, xmin=None, xmax=None, linear_bins=False, **kwargs):
    if not xmax:
        xmax = max(data)
    if not xmin:
        xmin = min(data)
    if linear_bins:
        bins = range(int(xmin), int(xmax))
    else:
        log_min_size = np.log10(xmin)
        log_max_size = np.log10(xmax)
        number_of_bins = np.ceil((log_max_size-log_min_size)*10)
        bins=np.unique(
                np.floor(
                    np.logspace(
                        log_min_size, log_max_size, num=number_of_bins)))
    hist, edges = np.histogram(data, bins, density=True)
    return edges, hist

def plot_pdf(data, ax=None, linear_bins=False, **kwargs):
    edges, hist = pdf(data, linear_bins=linear_bins, **kwargs)
    bin_centers = (edges[1:]+edges[:-1])/2.0
    hist[hist==0] = np.nan
    print sum(hist)
    print np.sum(hist*np.diff(edges))
    if not ax:
        plt.plot(bin_centers, hist, 'bo', **kwargs)
        ax = plt.gca()
    else:
        ax.plot(bin_centers, hist, 'bo', **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax

def log_binning(list_x, list_y, bin_count=35):
    # the returned values are raw values, not logarithmic values
    size = len(list_x)
    log_min_size = np.log10(min(list_x))
    log_max_size = np.log10(max(list_x))
    number_of_bins = np.ceil((log_max_size-log_min_size)*10)
    bins_x = np.unique(np.floor(np.logspace(log_min_size, log_max_size, num=number_of_bins)))
    # max_x = np.log10(max(list_x)+1)
    # min_x = np.log10(min(drop_zeros(list_x)))
    # bins_x = np.logspace(min_x, max_x, num=bin_count)
    new_bin_meanx_x, new_bin_means_y = [], []
    count_x = np.histogram(list_x, bins_x)[0]
    count_x_weight = np.histogram(list_x, bins_x, weights=list_x)[0].astype(float)
    for index in xrange(len(bins_x)-1):
        if count_x[index] != 0:
            new_bin_meanx_x.append(count_x_weight[index]/count_x[index])
            range_min, range_max = bins_x[index], bins_x[index+1]
            sum_y = 0.0
            for i in xrange(size):
                key = list_x[i]
                if (key >= range_min) and (key < range_max):
                    sum_y += list_y[i]
            new_bin_means_y.append(sum_y/count_x[index])
    return new_bin_meanx_x, new_bin_means_y


def cut_lists(list_x, list_y, fit_start=-1, fit_end=-1):
    if fit_start != -1:
        new_x, new_y = [], []
        for index in xrange(len(list_x)):
            if list_x[index] >= fit_start:
                new_x.append(list_x[index])
                new_y.append(list_y[index])
        list_x, list_y = new_x, new_y
    if fit_end != -1:
        new_x, new_y = [], []
        for index in xrange(len(list_x)):
            if list_x[index] < fit_end:
                new_x.append(list_x[index])
                new_y.append(list_y[index])
        list_x, list_y = new_x, new_y
    return (list_x, list_y)


def extended(ax, x, y, **args):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_ext = np.linspace(xlim[0], xlim[1], 100)
    p = np.polyfit(x, y , deg=1)
    y_ext = np.poly1d(p)(x_ext)
    ax.plot(x_ext, y_ext, **args)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def log_fit_ls(list_x, list_y, fit_start=-1, fit_end=-1):
    list_x, list_y = cut_lists(list_x, list_y, fit_start, fit_end)
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
    return (list_x, np.power(10, logY_fit))
    # return logX, logY_fit


def log_fit_ml(list_x, list_y, fit_start=-1, fit_end=-1):
    # TODO
    list_x, list_y = cut_lists(list_x, list_y, fit_start, fit_end)
    X = np.asarray(list_x, dtype=float)
    Y = np.asarray(list_y, dtype=float)
    logX = np.log10(X)
    logY = np.log10(Y)


def log_fit_ks(list_x, list_y, fit_start=-1, fit_end=-1):
    # TODO
    list_x, list_y = cut_lists(list_x, list_y, fit_start, fit_end)
    X = np.asarray(list_x, dtype=float)
    Y = np.asarray(list_y, dtype=float)
    logX = np.log10(X)
    logY = np.log10(Y)


def pmd(data):
    counter = Counter(data)
    counter_sum = sum(counter.values())
    klist, plist = [], []
    for key in counter:
        klist.append(key)
        plist.append(float(counter.get(key))/counter_sum)
    return (klist, plist)


def power_law_fit_ls(data, name, bin_count=30, ax=None, color='r', **kwargs):
    # data = drop_zeros(data)
    klist, plist = pmd(data)
    # print klist
    # print plist
    if not ax:
        plt.scatter(klist, plist, c=color, s=30, alpha=0.4,marker='+', label='Raw '+name)
        ax = plt.gca()
    else:
        ax.scatter(klist, plist, c=color, s=30, alpha=0.4,marker='+', label='Raw '+name)
    kmeans, pmeans = log_binning(klist, plist, bin_count)
    ax.scatter(kmeans, pmeans, c=color, s=50, marker='o', label='Binned '+name)
    '''whole fitting'''
    fit_x, fit_y = log_fit_ls(kmeans, pmeans)
    ax.plot(fit_x, fit_y, c=color, linewidth=2, linestyle='--', label='Fitted '+name)

    '''Split fitting'''
    # fit_x, fit_y = log_fit_ls(kmeans, pmeans, -1, 10)
    # ax.plot(fit_x, fit_y, c=color, linewidth=2,linestyle='-', label='Fitted 1 '+name)
    # fit_x, fit_y = log_fit_ls(kmeans, pmeans, 10, 500)
    # ax.plot(fit_x, fit_y, c='r',linewidth=2, linestyle='-', label='Fitted 2 '+name)
    # fit_x, fit_y = log_fit_ls(kmeans, pmeans, 100, 200)
    # ax.plot(fit_x, fit_y, c='g',linewidth=2, linestyle='-', label='Fitted 3 '+name)

    return ax


def neibors_static(DG, node, neib='pre', direct='in', weight=False):
    if neib == 'suc':
        neibors = DG.successors(node)
    else:
        neibors = DG.predecessors(node)
    if direct == 'out':
        if weight:
            values = [DG.out_degree(n, weight='weight') for n in neibors]
        else:
            values = [DG.out_degree(n) for n in neibors]
    else:
        if weight:
            values = [DG.in_degree(n, weight='weight') for n in neibors]
        else:
            values = [DG.in_degree(n) for n in neibors]
    if len(values) != len(neibors):
        print 'aaaaa........'
    return float(sum(values))/len(neibors)


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
suc_in_d, suc_out_d, pre_in_d, pre_out_d = [], [], [], []
suc_in_s, suc_out_s, pre_in_s, pre_out_s = [], [], [], []

# s = 0.0
# c = 0
for node in DG.nodes():
    # print 'Degree: %s \t %d \t %d \t %d' %(node, DG.in_degree(node), DG.out_degree(node), DG.degree(node))
    # print 'Strength: %s \t %d \t %d \t %d' %(node, DG.in_degree(node, weight='weight'), DG.out_degree(node, weight='weight'), DG.degree(node, weight='weight'))
    in_d, out_d, in_s, out_s = DG.in_degree(node), DG.out_degree(node), DG.in_degree(node, weight='weight'), DG.out_degree(node, weight='weight')
    if in_d and out_d:
        indegree.append(in_d)
        outdegree.append(out_d)
        instrength.append(in_s)
        outstrength.append(out_s)
        # print 'node',node,'indegree', in_d, 'outdegree', out_d
        suc_in_d.append(neibors_static(DG, node, 'suc', 'in', False))
        suc_out_d.append(neibors_static(DG, node, 'suc', 'out', False))
        pre_in_d.append(neibors_static(DG, node, 'pre', 'in', False))
        pre_out_d.append(neibors_static(DG, node, 'pre', 'out', False))

        suc_in_s.append(neibors_static(DG, node, 'suc', 'in', True))
        suc_out_s.append(neibors_static(DG, node, 'suc', 'out', True))
        pre_in_s.append(neibors_static(DG, node, 'pre', 'in', True))
        pre_out_s.append(neibors_static(DG, node, 'pre', 'out', True))

print
    # if in_d == 1:
    #     print out_d
        # s += in_d
#         c += 1
# print s/c


def deg_str_fit(lista, l, xlabel, ylabel, bin_count=50):
    plt.clf()
    pdf = plt.gca()
    pdf = power_law_fit_ls(lista, l, bin_count, ax=pdf, color='b')
    # pdf = power_law_fit_ls(outstrength, 'outstrength', 30, ax=pdf, color='r')
    pdf.set_ylabel(ylabel)
    pdf.set_xlabel(xlabel)
    pdf.set_xscale("log")
    pdf.set_yscale("log")
    pdf.set_xlim(xmin=1)
    pdf.set_ylim(ymax=1)
    handles, labels = pdf.get_legend_handles_labels()
    leg = pdf.legend(handles, labels, loc=0)
    leg.draw_frame(False)
    plt.show()


# deg_str_fit(indegree,'indegree','k','p(k)', 50)
# deg_str_fit(outdegree,'outdegree','k','p(k)', 50)
# deg_str_fit(instrength,'instrength','s','p(s)', 70)
plot_pdf(instrength)
# deg_str_fit(outstrength,'outstrength','s','p(s)', 70)


def dependence(listx, listy, l, xlabel, ylabel, bin_count=30):
    plt.clf()
    plt.scatter(listx, listy, s=20, c='k', alpha=0.3, marker='.', label='raw '+l)
    ax = plt.gca()
    xmeans, ymeans = log_binning(listx, listy, bin_count)
    ax.scatter(xmeans, ymeans, s=50, c='b', marker='o', label='binned '+l)
    xfit, yfit = log_fit_ls(xmeans, ymeans)
    ax.plot(xfit, yfit, c='r', linewidth=2, linestyle='--', label='Fitted '+l)
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xmin=1)
    ax.set_ylim(ymin=1)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc=0)
    leg.draw_frame(True)
    plt.show()

# dependence(indegree, outdegree, '$k_o(k_i)$', 'indegree', 'outdegree', 50)
# dependence(outdegree, indegree, '$k_i(k_o)$', 'outdegree', 'indegree', 50)
# dependence(instrength, outstrength, '$s_o(s_i)$', 'instrength', 'outstrength', 50)
# dependence(outstrength, instrength, '$s_i(s_o)$', 'outstrength', 'instrength', 50)

# dependence(indegree, pre_in_d, '$k_{i}^{pre}(k_i)$', 'indegree', 'Avg. Indegree of predecessors', 50)
# dependence(indegree, pre_out_d, '$k_{o}^{pre}(k_i)$', 'indegree', 'Avg. Outdegree of predecessors', 50)
# dependence(indegree, suc_in_d, '$k_{i}^{suc}(k_i)$', 'indegree', 'Avg. Indegree of successors', 50)
# dependence(indegree, suc_out_d, '$k_{o}^{suc}(k_i)$', 'indegree', 'Avg. Outdegree of successors', 50)

# dependence(outdegree, pre_in_d, '$k_{i}^{pre}(k_o)$', 'outdegree', 'Avg. Indegree of predecessors', 50)
# dependence(outdegree, pre_out_d, '$k_{o}^{pre}(k_o)$', 'outdegree', 'Avg. Outdegree of predecessors', 50)
# dependence(outdegree, suc_in_d, '$k_{i}^{suc}(k_o)$', 'outdegree', 'Avg. Indegree of successors', 50)
# dependence(outdegree, suc_out_d, '$k_{o}^{suc}(k_o)$', 'outdegree', 'Avg. Outdegree of successors', 50)

# dependence(instrength, pre_in_s, '$s_{i}^{pre}(s_i)$', 'Instrength', 'Avg. instrength of predecessors', 50)
# dependence(instrength, pre_out_s, '$s_{o}^{pre}(s_i)$', 'Instrength', 'Avg. outstrength of predecessors', 50)
# dependence(instrength, suc_in_s, '$s_{i}^{suc}(s_i)$', 'Instrength', 'Avg. instrength of successors', 50)
# dependence(instrength, suc_out_s, '$s_{o}^{suc}(s_i)$', 'Instrength', 'Avg. outstrength of successors', 50)

# dependence(outstrength, pre_in_d, '$s_{i}^{pre}(s_o)$', 'Outstrength', 'Avg. instrength of predecessors', 50)
# dependence(outstrength, pre_out_d, '$s_{o}^{pre}(s_o)$', 'Outstrength', 'Avg. outstrength of predecessors', 50)
# dependence(outstrength, suc_in_d, '$s_{i}^{suc}(s_o)$', 'Outstrength', 'Avg. instrength of successors', 50)
# dependence(outstrength, suc_out_d, '$s_{o}^{suc}(s_o)$', 'Outstrength', 'Avg. outstrength of successors', 50)



# print 'pearson correlation of indegree and outdegree: %f' %(pearson(indegree, instrength))
# print 'pearson correlation of instrength and outstrength: %f' %(pearson(outdegree, outstrength))
#
# print 'radius: %d' %(radius(DG))
# print 'diameter: %d' %(diameter(DG))
# print 'eccentricity: %s' %(eccentricity(DG))
# print 'center: %s' %(center(DG))
# print 'periphery: %s' %(periphery(DG))
# print 'density: %s' %(density(DG))

# klist, plist = pmd(instrength)
# fit = powerlaw.Fit(instrength)
# print 'powerlaw lib fit'
# print fit.alpha
# figPDF = powerlaw.plot_pdf(instrength, color='b')
# powerlaw.plot_pdf(instrength, linear_bins=True, color='r', ax=figPDF)
# figPDF.scatter(klist, plist, c='k', s=50, alpha=0.4,marker='+', label='Raw')
plt.show()
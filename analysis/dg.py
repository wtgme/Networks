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
    size = len(list_x)
    max_x = np.log10(max(list_x)+1)
    min_x = np.log10(min(drop_zeros(list_x)))
    bins_x = np.logspace(min_x, max_x, num=bin_count)
    new_bin_meanx_x, new_bin_means_y = [], []
    count_x = np.histogram(list_x, bins_x)[0]
    count_x_weight = np.histogram(list_x, bins_x, weights=list_x)[0].astype(float)
    for index in xrange(bin_count-1):
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


def log_fit(list_x, list_y, fit_start=-1, fit_end=-1):
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
    if not ax:
        plt.scatter(klist, plist, c=color, s=50, alpha=0.5,marker='+', label='Raw '+name)
        ax = plt.gca()
    else:
        ax.scatter(klist, plist, c=color, s=50, alpha=0.5,marker='+', label='Raw '+name)
    kmeans, pmeans = log_binning(klist, plist, bin_count)
    ax.scatter(kmeans, pmeans, c=color, s=30, marker='o', label='Binned '+name)
    fit_x, fit_y = log_fit(kmeans, pmeans)
    ax.plot(fit_x, fit_y, c=color, linestyle='--', label='Fitted '+name)
    return ax


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
    # if in_d == 1:
    #     print out_d
        # s += in_d
#         c += 1
# print s/c


def deg_str_fit(lista, l, xlabel, ylabel, bin_count=30):
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

# deg_str_fit(indegree,'indegree','k','p(k)')
# deg_str_fit(outdegree,'outdegree','k','p(k)')
# deg_str_fit(instrength,'instrength','s','p(s)')
# deg_str_fit(outstrength,'outstrength','s','p(s)')


def dependence(lista, listb, l, xlabel, ylabel, bin_count=30):
    plt.clf()
    plt.scatter(lista, listb, s=20, c='k', alpha=0.3, marker='.', label='raw '+l)
    ax = plt.gca()
    xmeans, ymeans = log_binning(lista, listb, bin_count)
    ax.scatter(xmeans, ymeans, s=50, c='b', marker='o', label='binned '+l)
    xfit, yfit = log_fit(xmeans, ymeans, 2)
    ax.plot(xfit, yfit, c='r', linewidth=2, linestyle='--', label='Fitted '+l)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin=1)
    ax.set_ylim(ymin=1)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc=0)
    leg.draw_frame(True)
    plt.show()

# dependence(indegree, outdegree, '$k_o(k_i)$', 'indegree', 'outdegree')
dependence(outdegree, indegree, '$k_i(k_o)$', 'outdegree', 'indegree', 20)
# dependence(instrength, outstrength, '$s_o(s_i)$', 'instrength', 'outstrength')
# dependence(outstrength, instrength, '$s_i(s_o)$', 'outstrength', 'instrength', 35)

# print 'pearson correlation of indegree and outdegree: %f' %(pearson(indegree, instrength))
# print 'pearson correlation of instrength and outstrength: %f' %(pearson(outdegree, outstrength))
#
# print 'radius: %d' %(radius(DG))
# print 'diameter: %d' %(diameter(DG))
# print 'eccentricity: %s' %(eccentricity(DG))
# print 'center: %s' %(center(DG))
# print 'periphery: %s' %(periphery(DG))
# print 'density: %s' %(density(DG))


# using the powerlaw lib
# def power_law_fit(list_x, label_x, savename='figure', list_y = None, Label_y = '', ax=None, **kwargs):
#     if not ax:
#         plt.plot(list_x, log_fit(list_x,list_y), **kwargs)
#         ax = plt.gca()
#     else:
#         ax.plot(list_x, log_fit(list_x,list_y), **kwargs)
#     plt.clf()
#     fit = powerlaw.Fit(list_x, discrete=True)
#     figPDF = fit.plot_pdf(color='b', linewidth=2, label=r"Empirical, "+label_x)
#     fit.power_law.plot_pdf(color='b', linestyle='--', ax=figPDF, label=r"Fit, "+label_x)
#     print 'alpha:', fit.power_law.alpha
#     print 'error:', fit.power_law.sigma
#
#     if list_y != None:
#         fit = powerlaw.Fit(list_y, discrete=True)
#         fit.plot_pdf(color='r', linewidth=2, ax=figPDF, label=r"Empirical, "+Label_y)
#         fit.power_law.plot_pdf(color='r', linestyle='--', ax=figPDF, label=r"Fit, "+Label_y)
#         print 'alpha:', fit.power_law.alpha
#         print 'error:', fit.power_law.sigma
#
#     figPDF.set_ylabel("p(k)")
#     figPDF.set_xlabel("k")
#     handles, labels = figPDF.get_legend_handles_labels()
#     leg = figPDF.legend(handles, labels, loc=3)
#     leg.draw_frame(False)
#     plt.show()
#     # plt.savefig(savename+'.eps', bbox_inches='tight')

'''Power-law Fitting'''
# power_law_fit(indegree, 'in-degree', outdegree, 'out-degree', 'degreepdf1')
# power_law_fit(indegree, 'in-degree', 'degreepdf1',outdegree,'out-degree')
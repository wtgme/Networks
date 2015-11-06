# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:40:27 2015

@author: tw5n14
"""
#import numpy as np
#import matplotlib.pyplot as plt
#
#x = [1,2,3,4]
#y = [3,5,7,10] # 10, not 9, so the fit isn't perfect
#
#fit = np.polyfit(x,y,1)
#fit_fn = np.poly1d(fit) 
#
## fit_fn is now a function which takes in x and returns an estimate for y
#
#plt.plot(x,y, 'yo')
#print fit_fn[1], fit_fn[0]
#plt.plot(x, fit_fn(x), '--k')
#plt.xlim(0, 5)
#plt.ylim(0, 12)
#plt.show()
#
#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
#print slope, intercept

#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#
#def func(x, a, b, c):
#    return a * np.exp(-b * x) + c
#
#x = np.linspace(0,4,50)
#y = func(x, 2.5, 1.3, 0.5)
#yn = y + 0.2*np.random.normal(size=len(x))
#
#popt, pcov = curve_fit(func, x, yn)
#
#plt.figure()
#plt.xscale('log')
#plt.yscale('log')
#plt.plot(x, yn, 'ko', label="Original Noised Data")
#plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
#plt.legend()
#plt.show()

# a = numpy.asarray([1,3,4,5], dtype=float)
# b = numpy.asarray([2,6,8,10], dtype=float)
# logA = numpy.log10(a)
# logB = numpy.log10(b)
# coefficients = numpy.polyfit(logB, logA, 1)
# polynomial = numpy.poly1d(coefficients)
# ys = polynomial(b)
# plt.plot(logB, logA)
# plt.plot(b, ys)

#import numpy as np
##import pylab as pl
#import matplotlib.pyplot as plt
#
#x = np.linspace(0, 2*np.pi, 100)
#plt.plot(x, np.sin(x), "-x", label=u"sin")
#plt.plot(x, np.random.standard_normal(len(x)), 'o', label=u"rand")
#leg = plt.legend(numpoints=3)
#
#plt.show()


#import numpy as np
#import matplotlib.pyplot as plt
#
#fig, ax1 = plt.subplots()
#t = np.arange(0.01, 10.0, 0.01)
#s1 = np.exp(t)
#ax1.plot(t, s1, 'b-')
#ax1.set_xlabel('time (s)')
## Make the y-axis label and tick labels match the line color.
#ax1.set_ylabel('exp', color='b')
#for tl in ax1.get_yticklabels():
#    tl.set_color('b')
#
#
#ax2 = ax1.twinx()
#s2 = np.sin(2*np.pi*t)
#ax2.plot(t, s2, 'r.')
#ax2.set_ylabel('sin', color='r')
#for tl in ax2.get_yticklabels():
#    tl.set_color('r')
#plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# plt.gca().set_color_cycle(None)
#
# for i in range(6):
#     plt.plot(np.arange(10, 1, -1) + i)
#
# plt.show()


# '''Maximum Likelihood'''
# # import the packages
# import numpy as np
# from scipy.optimize import minimize
# import scipy.stats as stats
#
# # Set up your x values
# x = np.linspace(0, 100, num=100)
#
# # Set up your observed y values with a known slope (2.4), intercept (5), and sd (4)
# yObs = 5 + 2.4*x + np.random.normal(0, 4, 100)
#
# # Define the likelihood function where params is a list of initial parameter estimates
# def regressLL(params, x, yObs):
#     # Resave the initial parameter guesses
#     b0 = params[0]
#     b1 = params[1]
#     sd = params[2]
#
#
#     # Calculate the predicted values from the initial parameter guesses
#     yPred = b0 + b1*x
#
#     # Calculate the negative log-likelihood as the negative sum of the log of a normal
#     # PDF where the observed values are normally distributed around the mean (yPred)
#     # with a standard deviation of sd
#     logLik = -np.sum(stats.norm.logpdf(yObs, loc=yPred, scale=sd))
#
#     # Tell the function to return the NLL (this is what will be minimized)
#     return(logLik)
#
# def ml(x, yObs):
#     # Make a list of initial parameter guesses (b0, b1, sd)
#     initParams = [1, 1, 1]
#
#     # Run the minimizer
#     results = minimize(regressLL, initParams, method='nelder-mead')
#
#     # Print the results. They should be really close to your actual values
#     print results.x

import networkx as nx
G = nx.DiGraph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
G.add_edge(0, 2)
G.add_edge(0, 1)
G.add_edge(0, 3)
G.add_edge(1, 3)
G.add_edge(1,0)

print G.predecessors(0)
print G.predecessors(2)
print G.successors(2)

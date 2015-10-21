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

a = numpy.asarray(a, dtype=float)
b = numpy.asarray(b, dtype=float)
logA = numpy.log10(a)
logB = numpy.log10(b)
coefficients = numpy.polyfit(logB, logA, 1)
polynomial = numpy.poly1d(coefficients)
ys = polynomial(b)
plt.plot(logB, logA)
plt.plot(b, ys)


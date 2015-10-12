import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

m = 3
N = 900

G = nx.barabasi_albert_graph(N, m)

degree_list=nx.degree(G).values()

kmin=min(degree_list)
kmax=max(degree_list)

bins=[float(k-0.5) for k in range(kmin,kmax+2,1)]
density, binedges = np.histogram(degree_list, bins=bins, density=True)
bins = np.delete(bins, -1)

logBins = np.logspace(np.log10(kmin), np.log10(kmax),num=20)
logBinDensity, binedges = np.histogram(degree_list, bins=logBins, density=True)
logBins = np.delete(logBins, -1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')

plt.plot(bins,density,'x',color='black')
plt.plot(logBins,logBinDensity,'x',color='blue')
plt.show()
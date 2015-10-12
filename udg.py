# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 16:22:26 2015

@author: tw5n14
"""

from networkx import *
import matplotlib.pyplot as plt
import csv


G = Graph()
userps = {}  

#load user profile from file
with open('poi.csv', 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        userp = {}
        userp['screem_name'] = row[1]
        userp['datetime'] = row[2]
        userp['descrip'] = row[3]
        userp['location'] = row[7]
        userp['gender'] = row[8]
        userp['gw'] = row[9]
        userp['cw'] = row[10]
        userps[tokens[0]] = userp
        del userp 



#with open('poi.txt','r') as fo:
#    for line in fo.readlines():
#        tokens = line.split(',')
#        print tokens[0]
        userp = {}
        userp['screem_name'] = tokens[1]
        userp['datetime'] = tokens[2]
        userp['descrip'] = tokens[3]
#        userp['location'] = tokens[7]
#        userp['gender'] = tokens[8]
        userps[tokens[0]] = userp
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
        if b_type == 'reply-to':
            if (G.has_node(n1)) and (G.has_node(n2)) and (G.has_edge(n1, n2)):
                # print n1, n2, G.has_node(n1), G.has_node(n2), G.has_edge(n1,n2), weightv, G[n2]
                G[n1][n2]['weight'] += weightv
            else:
                G.add_edge(n1, n2, weight=weightv)
            
    
        
# pos = random_layout(G)
# pos = shell_layout(G)
# pos = spring_layout(G)
#pos = spectral_layout(G)
# draw(G, pos)
# plt.show()



#network analysis
print 'The number of nodes: %d' %(G.order())
print 'The number of nodes: %d' %(G.__len__())
print 'The number of nodes: %d' %(G.number_of_nodes())
print 'The number of edges: %d' %(G.size())
print 'The number of self-loop: %d' %(G.number_of_selfloops())


#print 'The plot of in-degree and out-degree of nodes'
#print 'Node \t In-degree \t Out-degree'
#indegree, outdegree = [],[]
#for node in G.nodes():
#    print '%s \t %d \t %d \t %d' %(node, G.in_degree(node, weight='weight'), G.out_degree(node, weight='weight'), G.degree(node, weight='weight'))
#    indegree.append(G.in_degree(node, weight='weight'))
#    outdegree.append(G.out_degree(node, weight='weight'))
#
#print 'number of nodes: %d' %(len(G.nodes()))
#plt.scatter(indegree, outdegree, alpha=0.5)
#plt.title('Plot of In-degree and Out-degree (reply)')
#plt.xlabel('In-degree')
#plt.ylabel('Out-degree')
#plt.xlim(xmin=0.0)
#plt.ylim(ymin=0.0)
#plot.show()

    

##plot the numbers of nodes with degree K
#plt.title('Numbers of Nodes with In-degree K Plot(mention)')
#plt.ylabel('Number of nodes')
#plt.xlabel('Degree')
#degseq=list(G.in_degree(weight='weight').values())
#print degseq
#dmax=max(degseq)+1
#freq= [ 0 for d in range(dmax) ]
#for d in degseq:
#    freq[d] += 1
#plt.plot(freq)
##plt.plot(list(utils.cumulative_sum(freq)))
#plt.show()



##plot cumulative distribution of degree K
#plt.title('Cumulative Distribution of Nodes with Degree K Plot(mention)')
#plt.ylabel('P')
#plt.ylim(0.0,1.1)
#plt.xlabel('Degree')
#degseq=list(G.degree(weight='weight').values())
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

##histogram of path lengths
#print 'source vertex {taget: length,}'
#pathlengths = []
#for v in G.nodes():
#    spl = single_source_shortest_path_length(G, v)
##    print '%s %s' %(v, spl)
#    for p in spl.values():
#        pathlengths.append(p)
#print 'average shortest path length %s' %(sum(pathlengths)/len(pathlengths))
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

print (is_connected(G))

# print 'radius: %d' %(radius(G))
# print 'diameter: %d' %(diameter(G))
# print 'eccentricity: %s' %(eccentricity(G))
# print 'center: %s' %(center(G))
#print 'periphery: %s' %(periphery(G))
print 'density: %s' %(density(G))

#plt.title('Plot of Giant Component(reply)')
#pos = spring_layout(G)
#draw(G, pos, with_label=False, node_size=5)
#Gcc = sorted(connected_component_subgraphs(G), key = len, reverse = True)
#print 'number of components in network: %d' %(len(Gcc))
#G0 = Gcc[0]
#print 'the size of giant components: %d' %(G0.number_of_nodes())
#draw_networkx_edges(G0, pos, with_labels=False, edge_color='r', width = 1.0)
#for Gi in Gcc[1:]:
#    if len(Gi)>1:
#        draw_networkx_edges(Gi, pos, with_labels=False, edge_color='g', alpha=0.3,width=0.2)
#plt.show()


from networkx import *
import matplotlib.pyplot as plt

DG = nx.DiGraph()
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
        if b_type == 'mentioned':
            DG.add_edge(n1, n2, weight=weightv)
            # print n1, n2, weightv

        
# pos = nx.random_layout(DG)
# pos = nx.shell_layout(DG)
#pos = nx.spring_layout(DG)
#pos = nx.spectral_layout(DG)
#nx.draw(DG, pos)
#plt.show()
#pos = nx.spring_layout(DG)
# pos = nx.spectral_layout(DG)
#nx.draw(DG, pos)
#plt.show()


# load user profile from file
# with open('poi.txt','r') as fo:
#     for line in fo.readlines():
#         tokens = line.split(',')
#         user = User(tokens[0], tokens[1], tokens[2], tokens[3], tokens[7], tokens[8])
#         UserDic[user.uid] = user


#network analysis
print 'The number of nodes: %d' %(DG.order())
print 'The number of nodes: %d' %(DG.__len__())
print 'The number of nodes: %d' %(DG.number_of_nodes())
print 'The number of edges: %d' %(DG.size())
print 'The number of self-loop: %d' %(DG.number_of_selfloops())

#print 'The plot of in-degree and out-degree of nodes'
#print 'Node \t In-degree \t Out-degree'
#indegree, outdegree = [],[]
#for node in DG.nodes():
#    print '%s \t %d \t %d \t %d' %(node, DG.in_degree(node, weight='weight'), DG.out_degree(node, weight='weight'), DG.degree(node, weight='weight'))
#    indegree.append(DG.in_degree(node, weight='weight'))
#    outdegree.append(DG.out_degree(node, weight='weight'))
#
#print 'number of nodes: %d' %(len(DG.nodes()))
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
#degseq=list(DG.in_degree(weight='weight').values())
#print degseq
#dmax=max(degseq)+1
#freq= [ 0 for d in range(dmax) ]
#for d in degseq:
#    freq[d] += 1
#plt.plot(freq)
##plt.plot(list(nx.utils.cumulative_sum(freq)))
#plt.show()

##plot cumulative distribution of degree K
#plt.title('Cumulative Distribution of Nodes with Degree K Plot(mention)')
#plt.ylabel('P')
#plt.ylim(0.0,1.1)
#plt.xlabel('Degree')
#degseq=list(DG.degree(weight='weight').values())
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
#for v in DG.nodes():
#    spl = single_source_shortest_path_length(DG, v)
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

#print (is_connected(DG))

#print 'radius: %d' %(radius(DG))
#print 'diameter: %d' %(diameter(DG))
#print 'eccentricity: %s' %(eccentricity(DG))
#print 'center: %s' %(center(DG))
#print 'periphery: %s' %(periphery(DG))
#print 'density: %s' %(density(DG))

#nx.k_clique_communities(DG, 5)
#nx.draw(DG)
#plt.show()

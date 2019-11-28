

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

x2 = np.array([2, 4, 0, 1, 3, 0, 5])
x1 = np.array(list(range(len(x2))))
X = np.vstack((x1,x2)).T

labels = ['P{}'.format(i) for i in x1]

for i,(x1i,x2i) in enumerate(zip(x1,x2)):
    plt.scatter(x1i, x2i)
    plt.text(x1i+.03, x2i+.03, labels[i], fontsize=9)
plt.show()

#### Using Euclidean Distances

Z_ = linkage(pdist(X), metric='euclidean', method='average')
pd.DataFrame(Z_, columns=['P1','P2','Distance','Points in Cluster'])

#### Using Cosine Distances

distances = pdist(X,metric='cosine') # Returns the upper half matrix of the cosine distances matrix  
Z = linkage(y=distances, method='ward')
pd.DataFrame(Z, columns=['P1','P2','Distance','Points in Cluster'])

_, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
ax1.set_title('Cosine')
ax2.set_title('Euclidean')
_ = dendrogram(Z, ax=ax1)
_ = dendrogram(Z_, ax=ax2)

cluster_dict = defaultdict(list)
clusters = fcluster(Z_, 3, criterion='distance')
for i, c in enumerate(clusters):
    cluster_dict[c].append(i)
print(cluster_dict)



### Retrieve Cluster ID of each document

#### Assuming we know k --> Alternatives using distances etc

def same_dicts(d1,d2):
    return all([v1 == v2 for v1,v2 in zip(d1.values(), d2.values())])

def retrieve_doc_idx_by_level(cluster_idx, idx):
    ''' Return the indexes of the documents that match the cluster id '''
    doc_ids = np.array(list(range(len(cluster_idx))))
    mask = cluster_idx==idx
    return [c for c,i in zip(doc_ids,mask) if i]

def retrieve_hca_info(Z, criterion='maxclust', min_clusters=2, max_clusters=len(Z)//2+1):
    ''' Retrive the documents that belong to each cluster 
    for every merge done during the HCA process ''' 
    cluster_dict = defaultdict(lambda: defaultdict(list))
    for level in range(max_clusters, min_clusters-1, -1):
        cluster_idx = fcluster(Z, level, criterion=criterion)
        for c in range(1,1+level):
            cluster_dict[level][c].append(retrieve_doc_idx_by_level(cluster_idx,c))
    return cluster_dict

MIN_CLUSTERS = 2
MAX_CLUSTERS = 5
CRITERION = 'maxclust'
cluster_dict = retrieve_hca_info(Z_, CRITERION, MIN_CLUSTERS, MAX_CLUSTERS)

cluster_dict

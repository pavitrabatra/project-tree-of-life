import networkx as nx
import numpy as np
from entropy_on_networkx_graph import js_div
import matplotlib.pyplot as plt

folder="/entropy-of-networks/networkx_implementation/jensen_reordered"
#You may have to edit the folder path according to its location
data_array=[]
for i in range(1,19):
    d = np.loadtxt(f'{folder}/layer{i}.edges')
    data_array.append(d)


graph_array=[]
for i in range(18):
    g = nx.Graph(350)
    g.add_edges_from(data_array[i])
    graph_array.append(g)

beta = 0.1
D=np.zeros((18, 18))
for i in range(18):
    print(i)
    for j in range(18):
        print(j)
        D[i][j]=js_div(graph_array[i],beta,graph_array[j],beta)
        print(D[i][j])

fig = plt.figure()
ax1 = fig.subplots()
ax1.imshow(D)
ax1.invert_yaxis()
ax1.colorbar()
plt.show()


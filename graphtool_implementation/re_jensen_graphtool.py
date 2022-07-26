import graph_tool.all as gt
import numpy as np
from entropy_on_laplacian import js_div
import matplotlib.pyplot as plt

folder="/entropy-of-networks/graphtool_implementation/jensen_reordered"
#you may have to update the folder path according to folder location
data_array = []
laplacian_array = []

for i in range(1,19):
    d = np.loadtxt(f'{folder}/layer{i}.edges')
    data_array.append(d)


graph_array=[]
for i in range(18):
    g = gt.Graph(directed=False)
    g.add_vertex(350)
    g.add_edge_list(data_array[i])
    graph_array.append(g)
    l = gt.laplacian(g)
    l = l.toarray()
    laplacian_array.append(l)

beta = 0.1
D=np.zeros((18, 18))
for i in range(18):
    print(i)
    for j in range(18):
        print(j)
        D[i][j]=js_div(laplacian_array[i],beta,laplacian_array[j],beta)
        print(D[i][j])

ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
yticks = ['18','9','7','2','4','6','11','1','13','10','5','15','14','16','8','17','3','12']
labels = ['Vgnlintrts','Postrrfmx','Midvagina','Bucalmucs','Krtnzdgngv','LRtnclcr','RRtnclcr','Antenrns','Stool','Rantcbttf','Lantcbtiffs','Sprgngvlpl','Sbgngvlplq','Throat','PaltnTnsis','Tongdorsum','Hardpalate','Saliva']
fig = plt.figure()
ax1 = fig.subplots()
ax1.imshow(D,cmap='YlOrBr')
ax1.invert_yaxis()
plt.xticks(ticks,labels,rotation='vertical')
plt.yticks(ticks,yticks)
#ax1.colorbar()
plt.show()


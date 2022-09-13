import numpy as np
import os

dirname = "/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Data/snap/treeoflife.interactomes"
savedirname = "/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Data/snap/treeoflife.interactomes.save"
indxdirname = "/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Data/snap/treeoflife.interactomes.indx"

if not os.path.exists(indxdirname):
    os.mkdir(indxdirname)
if not os.path.exists(savedirname):
    os.mkdir(savedirname)

sizearray = []

for filename in os.listdir(dirname):
    print(filename)

    #loading interaction file; needs to be read as string
    l = np.loadtxt(f'{dirname}/{filename}', dtype=str)

    #generating a unique protein list
    l1 = l[:,0]
    l2 = l[:,1]
    ##concatenating the two lists
    ll = np.concatenate([l1,l2])
    ##getting a unique list out and creating a dictionary
    llu = np.unique(ll)
    lludict = {}
    for i in range(len(llu)):
        lludict[llu[i]] = i
    ### Alternative Slower Methods not creating dictionary
    ### llu = list(np.unique(ll))
    ### llu = np.unique(ll)

    #storing the network size
    N = len(llu)
    sizearray.append([filename, str(N)])

    #create the protein index file
    ind = np.arange(len(llu))
    index = list(zip(llu, ind))
    np.savetxt(f'{indxdirname}/{filename}', index, fmt='%s')

    #getting the interactions in readable formats
    interaction = []
    for i in range(len(l1)):
        prot1 = l[i,0]
        prot2 = l[i,1]
        interaction.append([lludict[prot1], lludict[prot2]])
        ## Alternative slower call/comparision methods
        ## interaction.append([llu.index(prot1), llu.index(prot2)])
        ## interaction.append([np.where(llu == prot1)[0][0], np.where(llu == prot2)[0][0]])

    ##create the readable interaction file
    np.savetxt(f'{savedirname}/{filename}', interaction, fmt='%d')
np.savetxt('size.txt', sizearray, fmt='%s')

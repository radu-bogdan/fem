import numpy as npy

def getIndices(liste, name):
    regions = npy.char.split(name,',').tolist()
    ind = npy.empty(shape=(0,),dtype=npy.int64)
    for k in regions:
        if k[0] == '*':
            n = npy.flatnonzero(npy.char.find(liste,k[1:])!=-1)
        else:
            n = npy.flatnonzero(npy.char.equal(liste,k))
        ind = npy.append(ind,n,axis=0)
    return npy.unique(ind)
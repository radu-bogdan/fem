import numpy as np
import scipy.sparse as sp

def makeProjectionMatrices(MESH,indices = np.empty(0, dtype=np.int)):
    
    noDOF = MESH.NoEdges
    
    
    jm = np.r_[0:2*noDOF]
    
    vek = np.r_[0:noDOF] + indices.size
    
    im1 = vek
    im2 = vek
    
    im1[indices + np.r_[0:indices.size]] = False
    im2[indices + np.r_[0:indices.size]-1] = False
    
    im = np.r_[im1,im2]
    
    vm = 1/2*np.ones((2,noDOF))
    vm[:,indices] = 1/np.sqrt(2)
    
    P = sparse(im.flatten(),jm.flatten(),vm.flatten(),noDOF+indices.size,2*noDOF)
    return P


def sparse(i, j, v, m, n):
    return sp.csc_matrix((v.flatten(), (i.flatten(), j.flatten())), shape = (m, n))


# function P = assem_RT0_to_BDM1(MESH,indices)

# % noDOF = MESH.RT0.noDOF;
# % 
# % im = [1:2:2*noDOF;
# %       2:2:2*noDOF];
# % 
# % jm = [1:noDOF;
# %       1:noDOF];
# % 
# % vm = ones(2,noDOF);
# % 
# % E = sparse(im(:),jm(:),vm(:),2*noDOF,noDOF);

# if nargin == 1
#     indices = [];
# end

# % indices =


# noDOF = MESH.RT0.noDOF;

# jm = [1:2:2*noDOF;
#       2:2:2*noDOF];

# % im = [1:noDOF;
# %       1:noDOF];
  
# vek = 1:noDOF + length(indices);
# im1 = vek;
# im2 = vek;
# im1(:,indices+(1:length(indices))) = [];
# im2(:,indices+(1:length(indices))-1) = [];

# im = [im1;im2];

# vm = 1/2*ones(2,noDOF);
# vm(:,indices) = 1/sqrt(2);

# P = sparse(im(:),jm(:),vm(:),noDOF+length(indices),2*noDOF);

# end
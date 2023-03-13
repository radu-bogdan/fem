#!/usr/bin/python --relpath_append ../

import sys
sys.path.insert(0,'../../') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import geometries
import MaterialLaws
import nonlinear_Algorithms

import plotly.io as pio
pio.renderers.default = 'browser'

np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = np.inf)
np.set_printoptions(precision = 8)

gmsh.initialize()
gmsh.model.add("Capacitor plates")
geometries.geometryP2()
gmsh.option.setNumber("Mesh.Algorithm", 2)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.MeshSizeMax",0.2)
gmsh.option.setNumber("Mesh.MeshSizeMin",0.2)

# gmsh.fltk.run()
# quit()

p,e,t,q = pde.petq_generate()
gmsh.clear()
gmsh.finalize()
#====================================================================================
MESH = pde.mesh(p,e,t,q)
#====================================================================================
# BASIS = pde.basis()
# LISTS = pde.lists(MESH)


#====================================================================================
#RHS J
f1 = lambda x,y : -700000+0*x
f2 = lambda x,y :  700000+0*x
#MaterialParameters
aIron=1829084.46555680
bIron=2.30000000000000
nuAir = 795775.715459477+0.0000001#cant be same as nu0 otherwise singular
nuIron_= 795775.715459477
nu0_=nuIron_;
nuIron = lambda x,y : nuIron_ + 0*x +0*y
nu0=  lambda x,y :nu0_ + 0*x +0*y
eins=lambda x,y :1.+ 0*x +0*y
ironsmall=520;

penalty = 10**10
#====================================================================================
#Vacuum reluctivity
Co = pde.int.evaluate(MESH, order = 0, coeff = nu0, regions = np.r_[1,2,3,4,5,6,7,8])
nu_aus = Co.diagonal()
#====================================================================================
#FemMatrices:
D2 = pde.int.assemble(MESH, order = 2)
D0 = pde.int.assemble(MESH, order = 0)
BM = pde.h1.assemble(MESH, space = 'P1', matrix = 'M', order = 2)
D = pde.l2.assemble(MESH, space = 'P0', matrix = 'M')
Db2 = pde.int.assembleB(MESH, order = 2)
BKx,BKy = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 0)
BD = pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = 0)

Kxx = BKx@D0@Co@BKx.T
Kyy = BKy@D0@Co@BKy.T
Mb = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = Kxx.shape)

B = Mb@Db2@Mb.T
Cx = BD @ D0 @ sps.diags(nu_aus) @ BKx.T
Cy = BD @ D0 @ sps.diags(nu_aus) @ BKy.T
M = BM@D2@BM.T
K = Kxx + Kyy + penalty*B
#====================================================================================

#====================================================================================
#RHS
CoF1 = pde.int.evaluate(MESH, order = 2, coeff = f1, regions = np.r_[7])
CoF2 = pde.int.evaluate(MESH, order = 2, coeff = f2, regions = np.r_[8])

M_f = BM@D2@(CoF1.diagonal()+CoF2.diagonal())
#====================================================================================

#====================================================================================
#support of the materials
supportIron = pde.int.evaluate(MESH, order = 0, coeff = eins, regions = np.r_[2,3])
supportAir= pde.int.evaluate(MESH, order = 0, coeff = eins, regions = np.r_[1,4,5,6,7,8])
#====================================================================================
#MaterialLaw Iron
eI,deI,ddeI = MaterialLaws.HerbertsMaterialE(aIron,bIron,nu0_,5)
gI,dgI,ddgI= MaterialLaws.HerbertsMaterialG(aIron,bIron,nu0_,5)

#LinearCase Iron
#eI,deI,ddeI= MaterialLaws.LinMaterialE(ironsmall,nu0_)
#gI,dgI,ddgI= MaterialLaws.LinMaterialG(ironsmall)

#MaterialLaw Air Linear
eA,deA,ddeA = MaterialLaws.LinMaterialE(nuAir,nu0_)
gA,dgA,ddgA= MaterialLaws.LinMaterialG(nuAir)

#getStarting solution
#====================================================================================
Alin=BKx@D0@(supportIron*ironsmall+supportAir*nuAir)@BKx.T+BKy@D0@(supportIron*nuAir+supportAir*nuAir)@BKy.T+penalty*B
u = sps.linalg.spsolve(Alin,M_f)
ulin=u
ux = BKx.T@u
uy = BKy.T@u

py=(uy-(supportAir@dgA(ux,uy)[1,:]+supportIron@dgI(ux,uy)[1,:])/nu0_)
px=(ux-(supportAir@dgA(ux,uy)[0,:]+supportIron@dgI(ux,uy)[0,:])/nu0_)
up=np.concatenate((u,px,py),axis=None)
#====================================================================================

def update_left(px,py):
    Dxx = D @ D0 @ sps.diags(nu_aus) @ sps.diags(1+ddeI(px,py)[0,0,:]@supportIron+ddeA(px,py)[0,0,:]@supportAir)@ D.T
    Dyy = D @ D0 @ sps.diags(nu_aus) @ sps.diags(1+ddeI(px,py)[1,1,:]@supportIron+ddeA(px,py)[1,1,:]@supportAir)@ D.T
    Dxy = D @ D0 @ sps.diags(nu_aus) @ sps.diags(ddeI(px,py)[1,0,:]@supportIron+ddeA(px,py)[1,0,:]@supportAir)@ D.T
    Dyx = D @ D0 @ sps.diags(nu_aus) @ sps.diags(ddeI(px,py)[0,1,:]@supportIron+ddeA(px,py)[0,1,:]@supportAir)@ D.T  
    return sps.vstack((sps.hstack((K,-Cx.T,-Cy.T)),
                       sps.hstack((-Cx, Dxx,Dxy)),
                       sps.hstack((-Cy, Dyx,Dyy))))
def extractp(up):
    return up[Kxx.shape[0]:Kxx.shape[0]+MESH.nt],up[Kxx.shape[0]+MESH.nt:up.shape[0]]

def extractu(up):
    return up[0:Kxx.shape[0]]

def femg(up):
    u=extractu(up)
    px,py=extractp(up)
    ux = BKx.T@u
    uy = BKy.T@u
    diffx=nu_aus*(ux-px)
    diffy=nu_aus*(uy-py)
    h1=BKx@D0@(-diffx.T)+BKy@D0@(-diffy.T)+M_f-penalty*B@u
    dex=supportAir@deA(px,py)[0,:]*nu_aus+supportIron@deI(px,py)[0,:]*nu_aus
    dey=supportAir@deA(px,py)[1,:]*nu_aus+supportIron@deI(px,py)[1,:]*nu_aus  
    l2x=D@D0@(diffx.T-dex.T)
    l2y=D@D0@(diffy.T-dey.T)
    rhs=np.concatenate((h1,l2x,l2y),axis=None);
    return -rhs

def fem_objective(up):    
    #gibt no ned
    ux = BKx.T@u
    uy = BKy.T@u 
    return 0
   
def femH(up):
    px,py=extractp(up)
    return update_left(px,py)

#=================================================================
#solve with Newton
up,flag=nonlinear_Algorithms.NewtonSparse(fem_objective,femg,femH,up)

#=================================================================
#check residuum of related problem
u=extractu(up)
ux = BKx.T@u
uy = BKy.T@u 
res=(-BKx@D0@(supportAir@dgA(ux,uy)[0,:].T+supportIron@dgI(ux,uy)[0,:].T) -BKy@D0@(supportAir@dgA(ux,uy)[1,:].T+supportIron@dgI(ux,uy)[1,:].T) + M_f -penalty*B@u)

print('Residuum in der Originalformulierung: ',np.linalg.norm(res))
#=================================================================

#=================================================================
#Plot
fig = MESH.pdesurf_hybrid(dict(trig = 'P1',quad = 'Q1', controls = 1), u)
fig.show()
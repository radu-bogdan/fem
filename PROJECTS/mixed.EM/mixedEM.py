import sys
sys.path.insert(0,'../../') # adds parent directory

import numpy as np
import gmsh
import pde
import scipy.sparse as sps
import scipy.sparse.linalg
import time
import plotly.io as pio
pio.renderers.default = 'browser'
# import nonlinear_Algorithms
import numba as nb
import pyamg

import matplotlib.pyplot as plt
import matplotlib
cmap = plt.cm.jet

##########################################################################################
# Loading mesh
##########################################################################################
# motor_stator_npz = np.load('meshes/motor_stator.npz', allow_pickle = True)

# p_stator = motor_stator_npz['p'].T
# e_stator = motor_stator_npz['e'].T
# t_stator = motor_stator_npz['t'].T
# q_stator = np.empty(0)
# regions_2d_stator = motor_stator_npz['regions_2d']
# regions_1d_stator = motor_stator_npz['regions_1d']
# m = motor_stator_npz['m']; m_new = m
# j3 = motor_stator_npz['j3']

# motor_rotor_npz = np.load('meshes/motor_rotor.npz', allow_pickle = True)

# p_rotor = motor_rotor_npz['p'].T
# e_rotor = motor_rotor_npz['e'].T
# t_rotor = motor_rotor_npz['t'].T
# q_rotor = np.empty(0)
# regions_2d_rotor = motor_rotor_npz['regions_2d']
# regions_1d_rotor = motor_rotor_npz['regions_1d']

motor_npz = np.load('meshes/motor.npz', allow_pickle = True)

p = motor_npz['p'].T
e = motor_npz['e'].T
t = motor_npz['t'].T
q = np.empty(0)
regions_2d = motor_npz['regions_2d']
regions_1d = motor_npz['regions_1d']
m = motor_npz['m']; m_new = m
j3 = motor_npz['j3']
##########################################################################################



##########################################################################################
# Parameters
##########################################################################################

nu0 = 10**7/(4*np.pi)

MESH = pde.mesh(p,e,t,q)
# MESH.refinemesh()
# MESH.refinemesh()
##########################################################################################



##########################################################################################
# Extract indices
##########################################################################################

mask_air_all = MESH.getIndices2d(regions_2d, 'air')
mask_stator_rotor_and_shaft = MESH.getIndices2d(regions_2d, 'iron')
mask_magnet = MESH.getIndices2d(regions_2d, 'magnet')
mask_coil = MESH.getIndices2d(regions_2d, 'coil')
mask_shaft = MESH.getIndices2d(regions_2d, 'shaft')
mask_iron_rotor = MESH.getIndices2d(regions_2d, 'rotor_iron', exact = 1)
mask_rotor_air = MESH.getIndices2d(regions_2d, 'rotor_air', exact = 1)
mask_air_gap_rotor = MESH.getIndices2d(regions_2d, 'air_gap_rotor', exact = 1)

mask_rotor     = mask_iron_rotor + mask_magnet + mask_rotor_air + mask_shaft
mask_linear    = mask_air_all + mask_magnet + mask_shaft + mask_coil
mask_nonlinear = mask_stator_rotor_and_shaft - mask_shaft

trig_rotor = MESH.t[np.where(mask_rotor)[0],0:3]
trig_air_gap_rotor = MESH.t[np.where(mask_air_gap_rotor)[0],0:3]
points_rotor = np.unique(trig_rotor)
points_rotor_and_airgaprotor = np.unique(np.r_[trig_rotor,trig_air_gap_rotor])


ind_stator_outer = np.flatnonzero(np.core.defchararray.find(list(regions_1d),'stator_outer')!=-1)
ind_rotor_outer = np.flatnonzero(np.core.defchararray.find(list(regions_1d),'rotor_outer')!=-1)
ind_edges_rotor_outer = np.where(np.isin(e[:,2],ind_rotor_outer))[0]
edges_rotor_outer = e[ind_edges_rotor_outer,0:2]

ind_trig_coils   = MESH.getIndices2d(regions_2d, 'coil', return_index = True)[0]
ind_trig_magnets = MESH.getIndices2d(regions_2d, 'magnet', return_index = True)[0]

R = lambda x: np.array([[np.cos(x),-np.sin(x)],
                        [np.sin(x), np.cos(x)]])

r1 = p[edges_rotor_outer[0,0],0]
a1 = 2*np.pi/edges_rotor_outer.shape[0]

# Adjust points on the outer rotor to be equally spaced.
for k in range(edges_rotor_outer.shape[0]):
    p[edges_rotor_outer[k,0],0] = r1*np.cos(a1*(k))
    p[edges_rotor_outer[k,0],1] = r1*np.sin(a1*(k))
##########################################################################################

    



H_KL = [0, 4.514672686, 16.55379985, 22.57336343,28.59292701,34.61249059, 37.62227239, 46.65161776, 46.65161776, 49.66139955, 52.67118134, 55.68096313, 58.69074492, 61.70052671, 64.7103085, 67.72009029, 70.72987208, 73.73965388, 79.75921746, 82.76899925, 88.78856283, 91.79834462, 97.8179082, 103.8374718, 112.8668172, 121.8961625, 130.9255079, 142.9646351, 151.9939804, 167.0428894, 182.0917983, 209.1798345, 236.2678706, 260.3461249, 293.4537246, 326.5613243, 371.7080512, 419.8645598, 474.0406321, 540.2558315, 615.5003762, 702.7840482, 811.1361926, 901.4296464, 1015.801354, 1202.407825, 1346.877351, 1455.229496, 1668.924003, 1840.481565, 2135.440181, 2340.105342, 2592.927013, 2749.435666, 3116.629044, 3279.157261, 3513.920241, 3784.800602, 4097.817908, 4362.678706, 4663.656885, 4814.145974, 5000.752445, 5241.534989, 5416.102333, 5602.708804, 5771.256584, 6075.244545, 6409.330324, 6752.445448, 7101.580135, 7513.920241, 7829.947329, 8158.013544, 8365.688488, 1.28203768e+04, 1.65447489e+04, 2.07163957e+04, 2.55500961e+04, 3.15206135e+04, 4.03204637e+04, 7.73038295e+04,
1.29272791e+05, 1.81241752e+05, 2.33210713e+05, 2.85179674e+05, 3.37148635e+05, 3.89117596e+05, 4.41086557e+05, 4.93055518e+05, 5.45024479e+05, 5.96993440e+05, 6.48962401e+05, 7.00931362e+05, 7.52900323e+05, 8.04869284e+05, 8.56838245e+05, 9.08807206e+05, 9.60776167e+05, 1.01274513e+06, 1.06471409e+06, 1.11668305e+06, 1.16865201e+06, 1.22062097e+06, 1.27258993e+06, 1.32455889e+06, 1.37652785e+06, 1.42849682e+06, 1.48046578e+06, 1.53243474e+06, 1.58440370e+06, 1.63637266e+06, 1.68834162e+06, 1.74031058e+06, 1.79227954e+06, 1.84424850e+06, 1.89621746e+06, 1.94818643e+06, 2.00015539e+06, 2.05212435e+06, 2.10409331e+06, 2.15606227e+06, 2.20803123e+06, 2.26000019e+06] + [(zz-5.0)/(4*np.pi*10**-7)+2.26000019e+06 for zz in np.linspace(5+0.0653, 10, 60)]

B_KL = [0, 0.007959134, 0.027873457, 0.043806714, 0.059739972, 0.083641356, 0.111528304, 0.151364444, 0.193197114, 0.235028284, 0.270883359, 0.304746402, 0.346577572, 0.392392807, 0.434223977, 0.482031243, 0.529838509, 0.577645775, 0.63341967, 0.685210999, 0.754929117, 0.812696543, 0.86050231, 0.908308077, 0.964080473, 1.03180506, 1.077617296, 1.11944397, 1.159280111, 1.19512919, 1.234962333, 1.270805416, 1.3066485, 1.328548859, 1.352436754, 1.372340585, 1.39223842, 1.410142725, 1.424059968, 1.439963247, 1.45586203, 1.471754817, 1.48564508, 1.49356824, 1.505463473, 1.523298829, 1.535179073, 1.541101208, 1.556931042, 1.566805764, 1.582595127, 1.59046133, 1.606271678, 1.610177799, 1.625931189, 1.629834312, 1.639677558, 1.649502816, 1.663291154, 1.669135347, 1.678945617, 1.680862704, 1.6887379, 1.694594084, 1.700483244, 1.704374376, 1.710266534, 1.718083273, 1.725885023, 1.733682276, 1.743468564, 1.749239311, 1.757050054, 1.764854802, 1.768735442, 1.86530612e+00, 1.93061224e+00, 1.99591837e+00, 2.06122449e+00, 2.12653061e+00, 2.19183673e+00, 2.25714286e+00, 2.32244898e+00, 2.38775510e+00, 2.45306122e+00, 2.51836735e+00, 2.58367347e+00, 2.64897959e+00, 2.71428571e+00, 2.77959184e+00, 2.84489796e+00, 2.91020408e+00, 2.97551020e+00, 3.04081633e+00, 3.10612245e+00, 3.17142857e+00, 3.23673469e+00, 3.30204082e+00, 3.36734694e+00, 3.43265306e+00, 3.49795918e+00, 3.56326531e+00, 3.62857143e+00, 3.69387755e+00, 3.75918367e+00, 3.82448980e+00, 3.88979592e+00, 3.95510204e+00, 4.02040816e+00, 4.08571429e+00, 4.15102041e+00, 4.21632653e+00, 4.28163265e+00, 4.34693878e+00, 4.41224490e+00, 4.47755102e+00, 4.54285714e+00, 4.60816327e+00, 4.67346939e+00, 4.73877551e+00, 4.80408163e+00, 4.86938776e+00, 4.93469388e+00, 5.00000000e+00] + [ zz for zz in np.linspace(5+0.0653, 10, 60)]

H_bosch = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 500000] + [(zz-2.46158548)/(4*np.pi*10**-7)+500000 for zz in np.linspace(2.46158548+(2.46158548-2.08458619), 10, 60)]
B_bosch = [0,  0.076361010000000,   0.151802010000000,   0.224483020000000,   0.294404020000000,   0.360645030000000,   0.422286030000000,   0.479327040000000,   0.532688040000000,   0.581449050000000,   0.626530050000000,   0.913580110000000,    1.047910160000000,   1.123360210000000,   1.172130270000000,   1.205260320000000,   1.231030370000000,  1.250360420000000,   1.266010480000000,   1.279820530000000,   1.356281060000000,   1.400541590000000,   1.436522120000000,   1.467902650000000,   1.497443190000000,   1.525143720000000,   1.551004250000000,   1.575944780000000,   1.599045310000000,   1.761970620000000,   1.836575930000000,   1.870701240000000,   1.891026550000000,   1.905831860000000,   1.919717170000000,   1.932682480000000,   1.945647790000000,   1.958613100000000,   2.084586190000000,   2.461585480000000] + [ zz for zz in np.linspace(2.46158548+(2.46158548-2.08458619), 10, 60)]


from scipy import interpolate

t, c, k = interpolate.splrep(np.array(H_KL), np.array(B_KL), s=0.1, k=3)
spline = interpolate.BSpline(t, c, k, extrapolate = True)
xx = np.linspace(0,1e7,10_000_000)
plt.cla(); plt.plot(xx,spline(xx))
print(spline(xx))

dxspline = spline.derivative()
plt.plot(xx,dxspline(xx))

dxxspline = dxspline.derivative()
plt.plot(xx,dxxspline(xx))

f = interpolate.interp1d(H_KL, B_KL)


##########################################################################################
# Assembling stuff
##########################################################################################

# u = np.zeros(MESH.np)

tm = time.monotonic()

phi_Hcurl = lambda x : pde.hcurl.assemble(MESH, space = 'NC1', matrix = 'phi', order = x)
# phi_Hdiv = lambda x : pde.hdiv.assemble(MESH, space = 'BDM1', matrix = 'phi', order = x)
curlphi_Hcurl = lambda x : pde.hcurl.assemble(MESH, space = 'NC1', matrix = 'curlphi', order = x)
phi_L2 = lambda x : pde.l2.assemble(MESH, space = 'P0', matrix = 'M', order = x)

D = lambda x : pde.int.assemble(MESH, order = x)

Mh = lambda x: phi_Hcurl(x)[0] @ D(x) @ phi_Hcurl(x)[0].T + \
               phi_Hcurl(x)[1] @ D(x) @ phi_Hcurl(x)[1].T

D1 = D(1); D2 = D(2); Mh1 = Mh(1); Mh2 = Mh(2)
phi_L2_o1 = phi_L2(1)
curlphi_Hcurl_o1 = curlphi_Hcurl(1)

phix_Hcurl_o1 = phi_Hcurl(1)[0];
phiy_Hcurl_o1 = phi_Hcurl(1)[1];

C = phi_L2_o1 @ D1 @ curlphi_Hcurl_o1.T

iMh = pde.tools.fastBlockInverse(Mh1)
S = C@iMh@C.T


M0 = 0; M1 = 0; M00 = 0; M10 = 0
for i in range(16):
    M0 += pde.int.evaluate(MESH, order = 1, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    M1 += pde.int.evaluate(MESH, order = 1, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    
    M00 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[0,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()
    M10 += pde.int.evaluate(MESH, order = 0, coeff = lambda x,y : m_new[1,i], regions = np.r_[ind_trig_magnets[i]]).diagonal()

aM = phix_Hcurl_o1@ D1 @(M0) +\
     phiy_Hcurl_o1@ D1 @(M1)

r = C@(iMh@aM)

x = sps.linalg.spsolve(S,r)




dphix_H1_o1, dphiy_H1_o1 = pde.h1.assemble(MESH, space = 'P1', matrix = 'K', order = 1)

Kxx = dphix_H1_o1 @ D1 @ dphix_H1_o1.T
Kyy = dphiy_H1_o1 @ D1 @ dphiy_H1_o1.T

phi_H1b = pde.h1.assembleB(MESH, space = 'P1', matrix = 'M', shape = dphix_H1_o1.shape, order = 1)

D_stator_outer = pde.int.evaluateB(MESH, order = 1, edges = ind_stator_outer)
B_stator_outer = phi_H1b@ D_stator_outer @ phi_H1b.T

aM = dphix_H1_o1@ D1 @(-M1) +\
     dphiy_H1_o1@ D1 @(+M0)
     
x2 = sps.linalg.spsolve(Kxx+Kyy+10**10*B_stator_outer,aM)
    
##########################################################################################
# Plotting stuff
##########################################################################################


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_aspect(aspect = 'equal')

ax1.cla()
MESH.pdegeom(ax = ax1)
# MESH.pdemesh2(ax = ax1)

Triang = matplotlib.tri.Triangulation(MESH.p[:,0], MESH.p[:,1], MESH.t[:,0:3])
chip = ax1.tripcolor(Triang, x, cmap = cmap, shading = 'flat', lw = 0.1)



ax2 = fig.add_subplot(122)
ax2.set_aspect(aspect = 'equal')

ax2.cla()
MESH.pdegeom(ax = ax2)
chip = ax2.tripcolor(Triang, x2, cmap = cmap, shading = 'gouraud', lw = 0.1)

fig.tight_layout()
fig.show()


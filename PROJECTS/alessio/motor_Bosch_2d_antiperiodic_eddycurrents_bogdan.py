from ngsolve import *
from netgen.csg import *
from math import pi
from scipy import interpolate
import ngsolve as ng
import numpy as np
import time
from ngsolve.solvers import *
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions
from ngsolve.internal import *
from netgen.occ import *
from netgen.meshing import IdentificationType
from collections import defaultdict
import scipy.sparse as sp
import scipy
import scipy.sparse.linalg as scspla

import sys
sys.path.append('')

buildMotor = True
bLinear = True
dampNewton = False
separated_mesh = True

def calculateCurrents(shift):
    I0peak = 1555.63491861 ### *1.5

    phase_shift_I1 = 0.0
    phase_shift_I2 = 2/3*pi#4/3*pi
    phase_shift_I3 = 4/3*pi#2/3*pi

    rotationAngle = 1.0*shift/points_airgap*cake_angle

    I1 = CoefficientFunction (I0peak * sin(polepairs*rotationAngle + phi0 + phase_shift_I1))    #U+
    I2 = CoefficientFunction ( (-1)* I0peak * sin( polepairs*rotationAngle + phi0 + phase_shift_I2))#V-
    I3 = CoefficientFunction (I0peak * sin(polepairs*rotationAngle + phi0 + phase_shift_I3))#W+

    cfJi = mesh.MaterialCF({area_coils_UPlus: I1* 2.75 / areaOfOneCoil, area_coils_VMinus: I2* 2.75 / areaOfOneCoil, area_coils_WPlus: I3* 2.75 / areaOfOneCoil, area_coils_UMinus: (-1)*I1* 2.75 / areaOfOneCoil, area_coils_VPlus: (-1)*I2* 2.75 / areaOfOneCoil, area_coils_WMinus: (-1)*I3* 2.75 / areaOfOneCoil}, default = 0)

    return cfJi

def getMasterSlaveMaps(shift):
    identications = mesh.ngmesh.GetIdentifications()
    # print("identications", identications)
    idces_side_array_dim = (len(identications),2)
    idces_side_array = np.zeros(idces_side_array_dim, dtype=int)
    for i in range(idces_side_array_dim[0]):
        idces_side_array[i][0] = identications[i][0]-1
        idces_side_array[i][1] = identications[i][1]-1

    # print("idces_side_array", idces_side_array)

    master_to_slave_periodic_map     = {}
    slave_to_master_periodic_map     = {}
    master_to_slave_antiperiodic_map = {}
    slave_to_master_antiperiodic_map = {}

    N = len(mesh.ngmesh.Points())
    M = idces_side_array.shape[0]

    for i in range(N):
        slave_to_master_antiperiodic_map[i] = i
        master_to_slave_antiperiodic_map[i] = i
        slave_to_master_periodic_map[i] = i
        master_to_slave_periodic_map[i] = i

    for k in range(M):
        idx_master = idces_side_array[k][0]
        idx_slave  = idces_side_array[k][1]
        slave_to_master_antiperiodic_map[idx_slave] = idx_master
        master_to_slave_antiperiodic_map[idx_master] = idx_slave

    if separated_mesh:
        idces_airgap_array, idces_corners_array = identifyPointsAirgap()

        M = idces_airgap_array.shape[0]

        for k in range(M):
            if (k+shift) > M:
                j = (k+shift)%(M)
                idx_master = idces_airgap_array[k][0]
                idx_slave  = idces_airgap_array[j][1]
                slave_to_master_antiperiodic_map[idx_slave] = idx_master
                master_to_slave_antiperiodic_map[idx_master] = idx_slave
                # slave_to_master_periodic_map[idx_slave] = idx_master
                # master_to_slave_periodic_map[idx_master] = idx_slave
            else:
                j = (k+shift)%(M)
                idx_master = idces_airgap_array[k][0]
                idx_slave  = idces_airgap_array[j][1]
                slave_to_master_periodic_map[idx_slave] = idx_master
                master_to_slave_periodic_map[idx_master] = idx_slave
                # slave_to_master_antiperiodic_map[idx_slave] = idx_master
                # master_to_slave_antiperiodic_map[idx_master] = idx_slave

        idx_corner_bottom_right = idces_corners_array[0][0]
        idx_corner_top_right    = idces_corners_array[0][1]
        idx_corner_bottom_left  = idces_corners_array[1][0]
        idx_corner_top_left     = idces_corners_array[1][1]

        # slave_to_master_antiperiodic_map[idx_corner_bottom_right] = idx_corner_bottom_right
        # master_to_slave_antiperiodic_map[idx_corner_bottom_right] = idx_corner_bottom_right
        # slave_to_master_antiperiodic_map[idx_corner_top_right] = idx_corner_top_right
        # master_to_slave_antiperiodic_map[idx_corner_top_right] = idx_corner_top_right
        # slave_to_master_antiperiodic_map[idx_corner_bottom_left] = idx_corner_bottom_left
        # master_to_slave_antiperiodic_map[idx_corner_bottom_left] = idx_corner_bottom_left
        # slave_to_master_antiperiodic_map[idx_corner_top_left] = idx_corner_top_left
        # master_to_slave_antiperiodic_map[idx_corner_top_left] = idx_corner_top_left

        # print("Before")
        # print("slave_to_master_antiperiodic_map[idx_corner_bottom_left]:  ", slave_to_master_antiperiodic_map[idx_corner_bottom_left])
        # print("slave_to_master_antiperiodic_map[idx_corner_top_left]:     ", slave_to_master_antiperiodic_map[idx_corner_top_left])
        print("idx_corner_bottom_right:  ", idx_corner_bottom_right)
        print("idx_corner_top_right:  ", idx_corner_top_right)
        print("idx_corner_bottom_left:  ", idx_corner_bottom_left)
        print("idx_corner_top_left:  ", idx_corner_top_left)
        # slave_to_master_antiperiodic_map[idx_corner_bottom_left] = idx_corner_bottom_left
        # slave_to_master_antiperiodic_map[idx_corner_top_left] = idx_corner_top_left
        # print("After")
        # print("slave_to_master_antiperiodic_map[idx_corner_bottom_left]:  ", slave_to_master_antiperiodic_map[idx_corner_bottom_left])
        # print("slave_to_master_antiperiodic_map[idx_corner_top_left]:     ", slave_to_master_antiperiodic_map[idx_corner_top_left])

        # slave_to_master_antiperiodic_map[idx_corner_bottom_right] = idx_corner_bottom_right
        # slave_to_master_antiperiodic_map[idx_corner_bottom_left] = idx_corner_bottom_right
        # slave_to_master_antiperiodic_map[idx_corner_top_left] = idx_corner_bottom_right
        # slave_to_master_periodic_map[idx_corner_top_right] = idx_corner_bottom_right

        # slave_to_master_antiperiodic_map[idx_corner_bottom_left] = idx_corner_bottom_left
        # slave_to_master_antiperiodic_map[idx_corner_top_left] = idx_corner_top_left

    else:
        idces_corners_array = idces_side_array


    return master_to_slave_antiperiodic_map, slave_to_master_antiperiodic_map, master_to_slave_periodic_map, slave_to_master_periodic_map, idces_corners_array

def moveNGmesh(displ, mesh):
    for p in mesh.ngmesh.Points():
        mip = mesh(p[0],p[1])
        v = displ(mip)
        p[0] += v[0]
        p[1] += v[1]
    mesh.ngmesh.Update()

def alpha(t):
    return 0.0 # rotationAngle * t / T     #t=T corresponds to 90deg rotation

def alphap(t):
    return 0.0 # rotationAngle / T     #t=T corresponds to 90deg rotation

def Equation1a(u,v):
    return cfNu * ( grad(u)[0]*grad(v)[0] + grad(u)[1]*grad(v)[1] ) * dx(definedon=mesh.Materials(area_air))

def Equation1b(u,v):
    if bLinear:
        return cfNu * ( grad(u)[0]*grad(v)[0] + grad(u)[1]*grad(v)[1] ) * dx(definedon=mesh.Materials(area_iron))
    else:
        return nuAca(sqrt( 1e-12 + grad(u)[0]* grad(u)[0] + grad(u)[1]*grad(u)[1] ) ) * ( grad(u)[0]*grad(v)[0] + grad(u)[1]*grad(v)[1] ) * dx(definedon=mesh.Materials(area_iron))

def Equation2(u,v):
    return ( cfNu * ( grad(u)[0]*grad(v)[0] + grad(u)[1]*grad(v)[1] ) - nuMagnet * BR * ( (magnetizationPerp_z_new[0]) * grad(v)[0] + magnetizationPerp_z_new[1] * grad(v)[1]) ) * dx(definedon=mesh.Materials(area_magnets))

def Equation3(u,v):
    return ( cfNu * ( grad(u)[0]*grad(v)[0] + grad(u)[1]*grad(v)[1] ) - cfJi * v ) * dx(definedon=mesh.Materials(area_iron))

def Equation2_withoutRHS(u,v):
    return cfNu * ( grad(u)[0]*grad(v)[0] + grad(u)[1]*grad(v)[1] ) * dx(definedon=mesh.Materials(area_magnets))

def Equation3_withoutRHS(u,v):
    return cfNu * ( grad(u)[0]*grad(v)[0] + grad(u)[1]*grad(v)[1] ) * dx(definedon=mesh.Materials(area_coils))


def SolveNewton(maxit=100, maxerr=1e-11, dampfactor=1, printing=False, callback=None, linesearch=False, printenergy=False, print_wrong_direction=False):
    w = gfu.vec.CreateVector()
    r = gfu.vec.CreateVector()
    uh = gfu.vec.CreateVector()

    e = BilinearForm(fes,symmetric = False)
    e += Variation( HB_density_Aca(sqrt(1e-12 + grad(u)*grad(u)))*dx(definedon=mesh.Materials(area_iron)) )
    e += Variation( 0.5*cfNu*grad(u)*grad(u)*dx(definedon = ~mesh.Materials(area_iron)))
    e += Variation( -cfJi*u*dx(definedon=mesh.Materials(area_coils)) )
    e += Variation( -nuMagnet*BR*((magnetizationPerp_z_new[0])*grad(u)[0] + magnetizationPerp_z_new[1]*grad(u)[1])*dx(definedon=mesh.Materials(area_magnets)) )

    numit = 0
    err = 1.
    for it in range(maxit):
        numit += 1
        if printing:
            print("Newton iteration ", it)
            if printenergy:
                print("Energy: ", e.Energy(gfu.vec))

        e.AssembleLinearization(gfu.vec)
        e.Apply(gfu.vec, r)

        if e.condense:
            print("Error! Condensed bilinear form not treated in NewtonGMRES")
        else:
            solveLinearSystem(e, r, w)

        err2 = InnerProduct(w,r)
        if print_wrong_direction:
            if err2 < 0:
                print("wrong direction")
        err = sqrt(abs(err2))
        if printing:
            print("err = ", err)

        tau = min(1, numit*dampfactor)

        if linesearch:
            uh.data = gfu.vec - tau*w
            energy = e.Energy(gfu.vec)
            while e.Energy(uh) > energy+(max(1e-14*abs(energy),maxerr)) and tau > 1e-10:
                tau *= 0.5
                uh.data = gfu.vec - tau * w
                if printing:
                    print ("tau = ", tau)
                    print ("energy uh = ", e.Energy(uh))
            gfu.vec.data = uh

        else:
            gfu.vec.data -= tau * w
        if callback is not None:
            callback(it, err)
        if abs(err) < maxerr: break
    else:
        print("Warning: Newton might not converge! Error = ", err)
        return (-1,numit)
    return (0,numit)

def solveStateEquation():
    if bLinear:
        rhs = LinearForm(fes)
        rhs += cfJi * v * dx(definedon=mesh.Materials(area_coils))
        rhs += ( nuMagnet * BR * ( (magnetizationPerp_z_new[0]) * grad(v)[0] + magnetizationPerp_z_new[1] * grad(v)[1]) ) * dx(definedon=mesh.Materials(area_magnets))
        rhs.Assemble()

        solveLinearSystem(a, rhs.vec, gfu.vec)
    else:
        if dampNewton:
            SolveNewton(maxit=1000, dampfactor=0.05, maxerr=1e-4, linesearch=False, printing=False)
        else:
            SolveNewton(maxit=1000, dampfactor=1.0, maxerr=1e-8, linesearch=True, printing=True)

def solveLinearSystem(a, rhs, x):
    for r in range(len(rhs)):
        mapped_antiperiodic_r = slave_to_master_antiperiodic_map[r]
        mapped_periodic_r = slave_to_master_periodic_map[r]

        if mapped_antiperiodic_r != r:
            rhs[mapped_antiperiodic_r] = rhs[mapped_antiperiodic_r] - rhs[r]
            rhs[r] = 0.

        if mapped_periodic_r != r:
            rhs[mapped_periodic_r] = rhs[mapped_periodic_r] + rhs[r]
            rhs[r] = 0.

    rowsA, colsA, valsA = a.mat.COO()

    A_dict = {}
    for entry in range(len(valsA)):
        A_dict[rowsA[entry], colsA[entry]] = valsA[entry]

    idces_corner_bottom_right = idces_corners_array[0][0]
    idces_corner_top_right    = idces_corners_array[0][1]
    idces_corner_bottom_left  = idces_corners_array[1][0]
    idces_corner_top_left     = idces_corners_array[1][1]

    for i in range(len(rowsA)):
        r = rowsA[i]
        c = colsA[i]
        v = valsA[i]

        mapped_antiperiodic_r = slave_to_master_antiperiodic_map[r]
        mapped_antiperiodic_c = slave_to_master_antiperiodic_map[c]
        mapped_periodic_r = slave_to_master_periodic_map[r]
        mapped_periodic_c = slave_to_master_periodic_map[c]

        if mapped_periodic_r != r and mapped_periodic_c != c:
            if (mapped_periodic_r, mapped_periodic_c) in A_dict:
                A_dict[mapped_periodic_r, mapped_periodic_c] = A_dict[mapped_periodic_r, mapped_periodic_c] + A_dict[r, c]
                A_dict[r, c] = 0.
            else:
                A_dict[mapped_periodic_r, mapped_periodic_c] = A_dict[r, c]
                A_dict[r, c] = 0.
        elif mapped_periodic_r != r:
            if (mapped_periodic_r, mapped_periodic_c) in A_dict:
                A_dict[mapped_periodic_r, mapped_periodic_c] = A_dict[mapped_periodic_r, mapped_periodic_c] + A_dict[r, c]
                A_dict[r, c] = 0.
            else:
                A_dict[mapped_periodic_r, mapped_periodic_c] = A_dict[r, c]
                A_dict[r, c] = 0.
        elif mapped_periodic_c != c:
            if (mapped_periodic_r, mapped_periodic_c) in A_dict:
                A_dict[mapped_periodic_r, mapped_periodic_c] = A_dict[mapped_periodic_r, mapped_periodic_c] + A_dict[r, c]
                A_dict[r, c] = 0.
            else:
                A_dict[mapped_periodic_r, mapped_periodic_c] = A_dict[r, c]
                A_dict[r, c] = 0.

        if mapped_antiperiodic_r != r and mapped_antiperiodic_c != c:
            if (mapped_antiperiodic_r, mapped_antiperiodic_c) in A_dict:
                A_dict[mapped_antiperiodic_r, mapped_antiperiodic_c] = A_dict[mapped_antiperiodic_r, mapped_antiperiodic_c] + A_dict[r, c]
                A_dict[r, c] = 0.
            else:
                A_dict[mapped_antiperiodic_r, mapped_antiperiodic_c] = A_dict[r, c]
                A_dict[r, c] = 0.
        elif mapped_antiperiodic_r != r:
            if (mapped_antiperiodic_r, mapped_antiperiodic_c) in A_dict:
                A_dict[mapped_antiperiodic_r, mapped_antiperiodic_c] = A_dict[mapped_antiperiodic_r, mapped_antiperiodic_c] - A_dict[r, c]
                A_dict[r, c] = 0.
            else:
                A_dict[mapped_antiperiodic_r, mapped_antiperiodic_c] = -A_dict[r, c]
                A_dict[r, c] = 0.
        elif mapped_antiperiodic_c != c:
            if (mapped_antiperiodic_r, mapped_antiperiodic_c) in A_dict:
                A_dict[mapped_antiperiodic_r, mapped_antiperiodic_c] = A_dict[mapped_antiperiodic_r, mapped_antiperiodic_c] - A_dict[r, c]
                A_dict[r, c] = 0.
            else:
                A_dict[mapped_antiperiodic_r, mapped_antiperiodic_c] = -A_dict[r, c]
                A_dict[r, c] = 0.

    M = len(A_dict)
    zero_entries = 0
    for (r,c) in A_dict:
        if A_dict[r,c] == 0. :
            zero_entries += 1
    M = M - zero_entries

    rowsAnp = np.zeros(M, dtype=np.int32)
    colsAnp = np.zeros(M, dtype=np.int32)
    valsAnp = np.zeros(M, dtype=np.float32)

    i = 0
    for (r,c) in A_dict:
        if A_dict[r,c] != 0.:
            rowsAnp[i] = r
            colsAnp[i] = c
            valsAnp[i] = A_dict[r,c]
            i += 1

    aKMat = sp.csr_matrix((valsAnp,(rowsAnp,colsAnp)))

    Ndof = fes.ndof
    sol = np.zeros(Ndof)

    freedofsvec_list = []
    for i in range(Ndof):
        mapped_periodic_i = slave_to_master_periodic_map[i]
        mapped_antiperiodic_i = slave_to_master_antiperiodic_map[i]
        if fes.FreeDofs()[i] and (mapped_periodic_i == i) and (mapped_antiperiodic_i == i):
        # if fes.FreeDofs()[i] and (mapped_periodic_i == i) and (mapped_antiperiodic_i == i) and (i != idces_corner_top_left) and (i != idces_corner_bottom_left) and (i != idces_corner_top_right):
            freedofsvec_list.append(int(i))
    freedofsvec = np.array(freedofsvec_list).astype(int)

    aKMatF = aKMat[freedofsvec,:] [:,freedofsvec]

    sol[freedofsvec] = scspla.spsolve(aKMat[freedofsvec,:] [:,freedofsvec],rhs.FV().NumPy()[freedofsvec])

    # for n1 in range(Ndof):
    #     if False: #n1 == idces_corner_top_left:
    #         x[n1] = -sol[idces_corner_bottom_right]
    #     # elif n1 == idces_corner_top_right:
    #     #     x[n1] = sol[idces_corner_bottom_right]
    #     # elif n1 == idces_corner_bottom_left:
    #     #     x[n1] = -sol[idces_corner_bottom_right]
    #     elif n1 == slave_to_master_periodic_map[n1] and n1 == slave_to_master_antiperiodic_map[n1]:
    #         x[n1] = sol[n1]
    #     elif n1 == slave_to_master_antiperiodic_map[n1]:
    #         x[master_to_slave_periodic_map[slave_to_master_periodic_map[n1]]] = sol[slave_to_master_periodic_map[n1]]
    #     elif n1 == slave_to_master_periodic_map[n1]:
    #         x[master_to_slave_antiperiodic_map[n1]] = -sol[slave_to_master_antiperiodic_map[n1]]

    for n1 in range(Ndof):
        if n1 == slave_to_master_periodic_map[n1] and n1 == slave_to_master_antiperiodic_map[n1]:
            x[n1] = sol[n1]
        elif n1 == slave_to_master_antiperiodic_map[n1]:
            x[n1] = sol[slave_to_master_periodic_map[n1]]
        elif n1 == slave_to_master_periodic_map[n1]:
            x[n1] = -sol[slave_to_master_antiperiodic_map[n1]]

def identifyPointsAirgap():
    N = len(mesh.ngmesh.Points())

    sliding_inner_reduced_points = []
    sliding_inner_points = []

    for k in range(N):
        p = mesh.ngmesh.Points()[k+1]
        if rotor_outer_mask.vec.FV()[k] > 0.5:
            sliding_inner_reduced_points.append((p,k))

        if airgap_inner_mask.vec.FV()[k] > 0.5:
            sliding_inner_points.append((p,k))

    nsliding_reduced = len(sliding_inner_reduced_points)
    nsliding = len(sliding_inner_points)

    # print("sliding_inner_reduced_points", nsliding_reduced)
    # print("sliding_inner_points", nsliding)

    sliding_inner_reduced_points.sort(key=pointAngle)
    sliding_inner_points.sort(key=pointAngle)

    idces_corners_array = np.zeros((2,2), dtype=int)
    idces_corners_array[0][0] = sliding_inner_reduced_points[0][1]
    idces_corners_array[0][1] = sliding_inner_points[0][1]
    idces_corners_array[1][0] = sliding_inner_reduced_points[nsliding-1][1]
    idces_corners_array[1][1] = sliding_inner_points[nsliding-1][1]

    idces_corner_bottom_right = sliding_inner_reduced_points[0][1]
    idces_corner_top_right    = sliding_inner_points[0][1]
    idces_corner_bottom_left  = sliding_inner_reduced_points[nsliding-1][1]
    idces_corner_top_left = sliding_inner_points[nsliding-1][1]


    # idces_array_dim = (nsliding-2,2)
    # idces_airgap_array = np.zeros(idces_array_dim, dtype=int)
    #
    # for j in range(nsliding-2):
    #     i = j+1

    idces_array_dim = (nsliding-1,2)
    idces_airgap_array = np.zeros(idces_array_dim, dtype=int)

    for j in range(nsliding-1):
        i = j

        p = sliding_inner_reduced_points[i][0]
        p_idces = sliding_inner_reduced_points[i][1]
        q = sliding_inner_points[i][0]
        q_idces = sliding_inner_points[i][1]

        p[0] = r2*cos((i)/(nsliding-1)*cake_angle)
        p[1] = r2*sin((i)/(nsliding-1)*cake_angle)
        q[0] = r2_augmented*cos((i)/(nsliding-1)*cake_angle)
        q[1] = r2_augmented*sin((i)/(nsliding-1)*cake_angle)

        idces_airgap_array[j][0] = p_idces
        idces_airgap_array[j][1] = q_idces

        mesh.ngmesh.Update()

    return idces_airgap_array, idces_corners_array

def pointAngle(p):
    px = p[0][0]
    py = p[0][1]
    return np.arctan2(py,px)

def pointRadius(p):
    px = p[0][0]
    py = p[0][1]
    return sqrt(py**2+px**2)

if (buildMotor==True):
    

cake_angle = pi/4
points_airgap = 155

lz = 0.1795
rTorqueOuter = r7
rTorqueInner = r2

offset = 0
polepairs  = 4
gamma_correction_model = -30.0
gamma = 40.0
gamma_correction_timestep = -1
phi0 = (gamma + gamma_correction_model + gamma_correction_timestep * polepairs) * pi/180.0

area_magnets = ''
for i in range(16):
    area_magnets = area_magnets + '|magnet' + str(i + 1)

area_coils = ''
for i in range(48):
    area_coils = area_coils + '|coil' + str(i + 1)

area_air = 'air_gap|air_gap_stator|air_gap_rotor|air|rotor_air|stator_air'
area_iron = "rotor_iron|stator_iron|shaft_iron"

areaOfOneCoil = (Integrate(1, mesh, order = 10, definedon=mesh.Materials('coil1'))) * 2

area_coils_UPlus = 'coil'+str(f48(offset+1))+'|coil'+str(f48(offset+2))
area_coils_VMinus = 'coil'+str(f48(offset+3))+'|coil'+str(f48(offset+4))
area_coils_WPlus = 'coil'+str(f48(offset+5))+'|coil'+str(f48(offset+6))
area_coils_UMinus = 'coil'+str(f48(offset+7))+'|coil'+str(f48(offset+8))
area_coils_VPlus = 'coil'+str(f48(offset+9))+'|coil'+str(f48(offset+10))
area_coils_WMinus = 'coil'+str(f48(offset+11))+'|coil'+str(f48(offset+12))

for k in range(1,4):
    area_coils_UPlus = area_coils_UPlus + '|coil' + str(f48(k*12+offset+1) )
    area_coils_UPlus = area_coils_UPlus + '|coil' + str(f48(k*12+offset+2) )
    area_coils_VMinus = area_coils_VMinus + '|coil' + str(f48(k*12+offset+3) )
    area_coils_VMinus = area_coils_VMinus + '|coil' + str(f48(k*12+offset+4) )
    area_coils_WPlus = area_coils_WPlus + '|coil' + str(f48(k*12+offset+5) )
    area_coils_WPlus = area_coils_WPlus + '|coil' + str(f48(k*12+offset+6) )
    area_coils_UMinus = area_coils_UMinus + '|coil' + str(f48(k*12+offset+7) )
    area_coils_UMinus = area_coils_UMinus + '|coil' + str(f48(k*12+offset+8) )
    area_coils_VPlus = area_coils_VPlus + '|coil' + str(f48(k*12+offset+9) )
    area_coils_VPlus = area_coils_VPlus + '|coil' + str(f48(k*12+offset+10) )
    area_coils_WMinus = area_coils_WMinus + '|coil' + str(f48(k*12+offset+11) )
    area_coils_WMinus = area_coils_WMinus + '|coil' + str(f48(k*12+offset+12) )

cfDesign = mesh.MaterialCF({area_iron: 3, area_coils: 2, area_air: 0, area_magnets: 1}, default=0)

#-------------------------------------------------------------------------------
#BH Kurve

def nuAca(s):
    # mit par1 = 200, par2 = 0.0005, par3 = 7, nu0 = 600000 passt analytische Kurve besser zu Messdaten
    par1 = 200
    par2 = 0.001 #0.0005
    par3 = 6 #7
    nu0 = 1e7/(4*pi) #600000
    nuVal = nu0-(nu0-par1) *exp( (-1)*par2*( s )**par3)
    return nuVal

def nuAca_diff(s):
    par1 = 200
    par2 = 0.001
    par3 = 6
    nu0 = 1e7/(4*pi)
    nuval_diff = par2*par3*(nu0-par1)*exp( (-1)*par2*( s )**par3)*s**( par3 - 1)
    return nuval_diff

B_Aca = [0.001*i for i in range(11000)]
H_Aca = [nuAca(p)*p for p in B_Aca]
HB_curve_Aca = BSpline(2, [0]+[0]+list(B_Aca), [0]+list(H_Aca) )
HB_density_Aca = HB_curve_Aca.Integrate()

#######################################

sigmaAir = 0
sigmaMagnet = 1e5 #1e4 #1e6 #1e2
sigmaIron = 0 #1e7
sigmaCoil = 0
cfSigma = mesh.MaterialCF({area_iron: sigmaIron, area_coils: sigmaCoil, area_air: sigmaAir, area_magnets: sigmaMagnet}, default=0)

nuAir = 1e7/(4*pi)
nuMagnet = nuAir / 1.05
nuIron = nuAir / 5100
nuCoil = nuAir
cfNu = mesh.MaterialCF({area_iron: nuIron, area_coils: nuCoil, area_air: nuAir, area_magnets: nuMagnet}, default=0)
Draw(cfNu, mesh, 'cfNu')

BR = 1.05*1.158095238095238 #remanence flux density
#if noMag == True:
    #BR = 0

Mperp_mag1 = CoefficientFunction( (-0.507223091788922, 0.861814791678634) )
Mperp_mag2 = CoefficientFunction( (-0.250741225095427, 0.968054150364350) )
Mperp_mag3 = (-1)*CoefficientFunction( (-0.968055971101187, 0.250734195544481) )
Mperp_mag4 = (-1)*CoefficientFunction( (-0.861818474866413, 0.507216833690415) )
Mperp_mag5 = CoefficientFunction( (-0.861814791678634, -0.507223091788922) )
Mperp_mag6 = CoefficientFunction( (-0.968054150364350, -0.250741225095427) )
Mperp_mag7 = (-1)*CoefficientFunction( (-0.250734195544481, -0.968055971101187) )
Mperp_mag8 = (-1)*CoefficientFunction( (-0.507216833690415, -0.861818474866413) )
Mperp_mag9 = CoefficientFunction( (0.507223091788922, -0.861814791678634) )
Mperp_mag10 = CoefficientFunction( (0.250741225095427, -0.968054150364350) )
Mperp_mag11 = (-1)*CoefficientFunction( (0.968055971101187, -0.250734195544481) )
Mperp_mag12 = (-1)*CoefficientFunction( (0.861818474866413, -0.507216833690415) )
Mperp_mag13 = CoefficientFunction( (0.861814791678634, 0.507223091788922) )
Mperp_mag14 = CoefficientFunction( (0.968054150364350, 0.250741225095427) )
Mperp_mag15 = (-1)*CoefficientFunction( (0.250734195544481, 0.968055971101187) )
Mperp_mag16 = (-1)*CoefficientFunction( (0.507216833690415, 0.861818474866413) )

magnetizationPerp_z_new_1 = mesh.MaterialCF({'magnet1': Mperp_mag1[0], 'magnet2': Mperp_mag2[0], 'magnet3': Mperp_mag3[0], 'magnet4': Mperp_mag4[0], 'magnet5': Mperp_mag5[0], 'magnet6': Mperp_mag6[0], 'magnet7': Mperp_mag7[0], 'magnet8': Mperp_mag8[0], 'magnet9': Mperp_mag9[0], 'magnet10': Mperp_mag10[0], 'magnet11': Mperp_mag11[0], 'magnet12': Mperp_mag12[0], 'magnet13': Mperp_mag13[0], 'magnet14': Mperp_mag14[0], 'magnet15': Mperp_mag15[0], 'magnet16': Mperp_mag16[0] }, default = 0)

magnetizationPerp_z_new_2 = mesh.MaterialCF({'magnet1': Mperp_mag1[1], 'magnet2': Mperp_mag2[1], 'magnet3': Mperp_mag3[1], 'magnet4': Mperp_mag4[1], 'magnet5': Mperp_mag5[1], 'magnet6': Mperp_mag6[1], 'magnet7': Mperp_mag7[1], 'magnet8': Mperp_mag8[1], 'magnet9': Mperp_mag9[1], 'magnet10': Mperp_mag10[1], 'magnet11': Mperp_mag11[1], 'magnet12': Mperp_mag12[1], 'magnet13': Mperp_mag13[1], 'magnet14': Mperp_mag14[1], 'magnet15': Mperp_mag15[1], 'magnet16': Mperp_mag16[1] }, default = 0)

magnetizationPerp_z_new = CoefficientFunction( (magnetizationPerp_z_new_1, magnetizationPerp_z_new_2) )
magnetization_z_new = CoefficientFunction( (magnetizationPerp_z_new[1], -magnetizationPerp_z_new[0]) )

Draw(magnetizationPerp_z_new, mesh, "new_magnetizationPerp_z_new")
Draw(magnetization_z_new, mesh, "magnetization_z_new")
Draw(cfDesign,mesh,"cfDesign")
Draw(cfNu,mesh,"cfNu")

fes = H1(mesh=mesh, order=1, dirichlet='stator_outer')

airgap_inner_mask = GridFunction(fes)
rotor_outer_mask = GridFunction(fes)

airgap_inner_mask.Set(CoefficientFunction((1)), definedon = mesh.Boundaries("airgap_inner"))
rotor_outer_mask.Set(CoefficientFunction((1)), definedon = mesh.Boundaries("rotor_outer"))

area_magnet1 = Integrate(CoefficientFunction(1), mesh, order = 10, definedon=mesh.Materials('magnet1'))

Draw(airgap_inner_mask, mesh, 'airgap_inner_mask')
Draw(rotor_outer_mask, mesh, 'rotor_outer_mask')

u,v = fes.TnT()

N = len(mesh.ngmesh.Points())
points_airgap = 0
for k in range(N):
    if rotor_outer_mask.vec.FV()[k] > 0.5:
        points_airgap += 1

quarter_steps = points_airgap
n_steps = 9
shift_step = int(quarter_steps/2/n_steps)

gfu = GridFunction(fes)

cfJi = calculateCurrents(0)
master_to_slave_antiperiodic_map, slave_to_master_antiperiodic_map, master_to_slave_periodic_map, slave_to_master_periodic_map, idces_corners_array = getMasterSlaveMaps(0)

a = BilinearForm(fes)
a += Equation1a(u, v)
a += Equation1b(u, v)
a += Equation2_withoutRHS(u, v)
a += Equation3_withoutRHS(u, v)
a.Assemble()

solveStateEquation()

Draw(gfu,mesh,"gfu")

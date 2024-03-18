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

def drawMagnet1(k):
    m1xnew = m1[0]*cos(k*pi/4) -m1[1]*sin(k*pi/4)
    m1ynew = m1[0]*sin(k*pi/4) +m1[1]*cos(k*pi/4)
    m1new = (m1xnew,m1ynew,0)

    m2xnew = m2[0]*cos(k*pi/4) -m2[1]*sin(k*pi/4)
    m2ynew = m2[0]*sin(k*pi/4) +m2[1]*cos(k*pi/4)
    m2new = (m2xnew,m2ynew,0)

    m3xnew = m3[0]*cos(k*pi/4) -m3[1]*sin(k*pi/4)
    m3ynew = m3[0]*sin(k*pi/4) +m3[1]*cos(k*pi/4)
    m3new = (m3xnew,m3ynew,0)

    m4xnew = m4[0]*cos(k*pi/4) -m4[1]*sin(k*pi/4)
    m4ynew = m4[0]*sin(k*pi/4) +m4[1]*cos(k*pi/4)
    m4new = (m4xnew,m4ynew,0)

    a5xnew = a5[0]*cos(k*pi/4) -a5[1]*sin(k*pi/4)
    a5ynew = a5[0]*sin(k*pi/4) +a5[1]*cos(k*pi/4)
    a5new = (a5xnew,a5ynew,0)

    a6xnew = a6[0]*cos(k*pi/4) -a6[1]*sin(k*pi/4)
    a6ynew = a6[0]*sin(k*pi/4) +a6[1]*cos(k*pi/4)
    a6new = (a6xnew,a6ynew,0)

    a7xnew = a7[0]*cos(k*pi/4) -a7[1]*sin(k*pi/4)
    a7ynew = a7[0]*sin(k*pi/4) +a7[1]*cos(k*pi/4)
    a7new = (a7xnew,a7ynew,0)

    a8xnew = a8[0]*cos(k*pi/4) -a8[1]*sin(k*pi/4)
    a8ynew = a8[0]*sin(k*pi/4) +a8[1]*cos(k*pi/4)
    a8new = (a8xnew,a8ynew,0)

    #Draw magnet
    seg1 = Segment(m1new,m2new);
    seg2 = Segment(m2new,m3new);
    seg3 = Segment(m3new,m4new);
    seg4 = Segment(m4new,m1new);
    magnet1 = Face(Wire([seg1,seg2,seg3,seg4]))
    #Draw air around magnet
    air_seg1 = Segment(m1new,a5new)
    air_seg2 = Segment(a5new,a6new)
    air_seg3 = Segment(a6new,m2new)
    air_seg4 = Segment(m2new,m1new)
    air_magnet1_1 = Face(Wire([air_seg1,air_seg2,air_seg3,air_seg4]))
    air_seg5 = Segment(m4new,m3new)
    air_seg6 = Segment(m3new,a7new)
    air_seg7 = Segment(a7new,a8new)
    air_seg8 = Segment(a8new,m4new)
    air_magnet1_2 = Face(Wire([air_seg5,air_seg6,air_seg7,air_seg8]))

    return (magnet1,air_magnet1_1,air_magnet1_2)

def drawMagnet2(k):
    m5xnew = m5[0]*cos(k*pi/4) -m5[1]*sin(k*pi/4)
    m5ynew = m5[0]*sin(k*pi/4) +m5[1]*cos(k*pi/4)
    m5new = (m5xnew,m5ynew,0)

    m6xnew = m6[0]*cos(k*pi/4) -m6[1]*sin(k*pi/4)
    m6ynew = m6[0]*sin(k*pi/4) +m6[1]*cos(k*pi/4)
    m6new = (m6xnew,m6ynew,0)

    m7xnew = m7[0]*cos(k*pi/4) -m7[1]*sin(k*pi/4)
    m7ynew = m7[0]*sin(k*pi/4) +m7[1]*cos(k*pi/4)
    m7new = (m7xnew,m7ynew,0)

    m8xnew = m8[0]*cos(k*pi/4) -m8[1]*sin(k*pi/4)
    m8ynew = m8[0]*sin(k*pi/4) +m8[1]*cos(k*pi/4)
    m8new = (m8xnew,m8ynew,0)

    a1xnew = a1[0]*cos(k*pi/4) -a1[1]*sin(k*pi/4)
    a1ynew = a1[0]*sin(k*pi/4) +a1[1]*cos(k*pi/4)
    a1new = (a1xnew,a1ynew,0)

    a2xnew = a2[0]*cos(k*pi/4) -a2[1]*sin(k*pi/4)
    a2ynew = a2[0]*sin(k*pi/4) +a2[1]*cos(k*pi/4)
    a2new = (a2xnew,a2ynew,0)

    a3xnew = a3[0]*cos(k*pi/4) -a3[1]*sin(k*pi/4)
    a3ynew = a3[0]*sin(k*pi/4) +a3[1]*cos(k*pi/4)
    a3new = (a3xnew,a3ynew,0)

    a4xnew = a4[0]*cos(k*pi/4) -a4[1]*sin(k*pi/4)
    a4ynew = a4[0]*sin(k*pi/4) +a4[1]*cos(k*pi/4)
    a4new = (a4xnew,a4ynew,0)

    #Draw magnet
    seg1 = Segment(m5new,m6new);
    seg2 = Segment(m6new,m7new);
    seg3 = Segment(m7new,m8new);
    seg4 = Segment(m8new,m5new);
    magnet2 = Face(Wire([seg1,seg2,seg3,seg4]))
    air_seg1 = Segment(m5new,a3new)
    air_seg2 = Segment(a3new,a4new)
    air_seg3 = Segment(a4new,m6new)
    air_seg4 = Segment(m6new,m5new)
    air_magnet2_1 = Face(Wire([air_seg1,air_seg2,air_seg3,air_seg4]))
    air_seg5 = Segment(m8new,m7new)
    air_seg6 = Segment(m7new,a2new)
    air_seg7 = Segment(a2new,a1new)
    air_seg8 = Segment(a1new,m8new)
    air_magnet2_2 = Face(Wire([air_seg5,air_seg6,air_seg7,air_seg8]))

    return (magnet2,air_magnet2_1,air_magnet2_2)

def drawStatorNut(k):
    #Points for Stator Nut and air in the stator
    s1 = (79.04329892000*10**(-3),3.9538335974*10**(-3),0)
    s2 = (80.143057128*10**(-3),4.0037794254*10**(-3),0)
    s3 = (80.387321219*10**(-3),2.965459706*10**(-3),0)
    s4 = (98.78501315600001*10**(-3),3.9007973292*10**(-3),0)
    s5 = (98.44904989600001*10**(-3),9.026606148400001*10**(-3),0)
    s6 = (80.086666706*10**(-3),7.5525611543*10**(-3),0)
    s7 = (79.980020247*10**(-3),6.4912415424*10**(-3),0)
    s8 = (78.88229587*10**(-3),6.4102654448*10**(-3),0)

    s1xnew = s1[0]*cos(k*pi/24) -s1[1]*sin(k*pi/24)
    s1ynew = s1[0]*sin(k*pi/24) +s1[1]*cos(k*pi/24)
    s1new = (s1xnew,s1ynew,0)

    s2xnew = s2[0]*cos(k*pi/24) -s2[1]*sin(k*pi/24)
    s2ynew = s2[0]*sin(k*pi/24) +s2[1]*cos(k*pi/24)
    s2new = (s2xnew,s2ynew,0)

    s3xnew = s3[0]*cos(k*pi/24) -s3[1]*sin(k*pi/24)
    s3ynew = s3[0]*sin(k*pi/24) +s3[1]*cos(k*pi/24)
    s3new = (s3xnew,s3ynew,0)

    s4xnew = s4[0]*cos(k*pi/24) -s4[1]*sin(k*pi/24)
    s4ynew = s4[0]*sin(k*pi/24) +s4[1]*cos(k*pi/24)
    s4new = (s4xnew,s4ynew,0)

    s5xnew = s5[0]*cos(k*pi/24) -s5[1]*sin(k*pi/24)
    s5ynew = s5[0]*sin(k*pi/24) +s5[1]*cos(k*pi/24)
    s5new = (s5xnew,s5ynew,0)

    s6xnew = s6[0]*cos(k*pi/24) -s6[1]*sin(k*pi/24)
    s6ynew = s6[0]*sin(k*pi/24) +s6[1]*cos(k*pi/24)
    s6new = (s6xnew,s6ynew,0)

    s7xnew = s7[0]*cos(k*pi/24) -s7[1]*sin(k*pi/24)
    s7ynew = s7[0]*sin(k*pi/24) +s7[1]*cos(k*pi/24)
    s7new = (s7xnew,s7ynew,0)

    s8xnew = s8[0]*cos(k*pi/24) -s8[1]*sin(k*pi/24)
    s8ynew = s8[0]*sin(k*pi/24) +s8[1]*cos(k*pi/24)
    s8new = (s8xnew,s8ynew,0)

    #Draw stator coil
    seg1 = Segment(s2new,s3new)
    seg2 = Segment(s3new,s4new)
    seg3 = Segment(s4new,s5new)
    seg4 = Segment(s5new,s6new)
    seg5 = Segment(s6new,s7new)
    seg6 = Segment(s7new,s2new)
    stator_coil = Face(Wire([seg1,seg2,seg3,seg4,seg5,seg6]))
    #Draw air nut in the stator
    air_seg1 = Segment(s1new,s2new)
    air_seg2 = Segment(s2new,s7new)
    air_seg3 = Segment(s7new,s8new)
    air_seg4 = Segment(s8new,s1new)
    stator_air = Face(Wire([air_seg1,air_seg2,air_seg3,air_seg4]))

    stator_air = stator_air-(stator_air*air_gap)
    return (stator_coil,stator_air)

def f48(s):
    return (s-1)%48+1

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
    orign = (0,0);
    #inner radius rotor
    r1 = 26.5*10**(-3);
    # r1 = 10*10**(-3);
    #outer radius rotor
    r2_original = 78.23225*10**(-3);
    r2 = 78.23225*10**(-3); # 78.62725*10**(-3); #reduced
    r2_augmented = 78.236*10**(-3);
    # r2_original = 78.63225*10**(-3);
    # r2 = 78.63225*10**(-3); # 78.62725*10**(-3); #reduced
    # r2_augmented = 78.636*10**(-3);
    #sliding mesh rotor
    r4_reduced = 78.83*10**(-3);
    r4 = 78.8354999*10**(-3);
    #sliding mesh stator
    r6 = 79.03874999*10**(-3);
    #inner radius stator
    r7 = 79.242*10**(-3);
    #outer radius stator
    r8 = 116*10**(-3)

    #Points for magnet1 and air around magnet1
    m1 = (69.23112999*10**(-3),7.535512*10**(-3),0)
    m2 = (74.828958945*10**(-3),10.830092744*10**(-3),0)
    m3 = (66.13621099700001*10**(-3),25.599935335*10**(-3),0)
    m4 = (60.53713*10**(-3),22.30748*10**(-3),0)
    a5 = (69.75636*10**(-3),5.749913*10**(-3),0)
    a6 = (75.06735*10**(-3),3.810523*10**(-3),0)
    a7 = (65.3506200*10**(-3),26.51379*10**(-3),0)
    a8 = (59.942145092*10**(-3),24.083661604*10**(-3),0)
    #Points for magnet2 and air around magnet2
    m5 = (58.579985516*10**(-3), 27.032444757*10**(-3),0)
    m6 = (64.867251151*10**(-3),28.663475405*10**(-3),0)
    m7 = (60.570096319*10**(-3),45.254032279*10**(-3),0)
    m8 = (54.282213127*10**(-3),43.625389857*10**(-3),0)
    a1 = (53.39099766*10**(-3),45.259392713*10**(-3),0)
    a2 = (55.775078884*10**(-3),50.386185578*10**(-3),0)
    a3 = (59.41521771*10**(-3),25.355776837*10**(-3),0)
    a4 = (65.12210917100001*10**(-3),27.707477175*10**(-3),0)
    #Points for Stator Nut and air in the stator
    s1 = (79.04329892000*10**(-3),3.9538335974*10**(-3),0)
    s2 = (80.143057128*10**(-3),4.0037794254*10**(-3),0)
    s3 = (80.387321219*10**(-3),2.965459706*10**(-3),0)
    s4 = (98.78501315600001*10**(-3),3.9007973292*10**(-3),0)
    s5 = (98.44904989600001*10**(-3),9.026606148400001*10**(-3),0)
    s6 = (80.086666706*10**(-3),7.5525611543*10**(-3),0)
    s7 = (79.980020247*10**(-3),6.4912415424*10**(-3),0)
    s8 = (78.88229587*10**(-3),6.4102654448*10**(-3),0)

    domains = []

    h_max = 0.005
    #
    # h_air_gap = 2.0*(r6-r4) #0.05*h_max
    # h_air_magnets = h_max
    # h_coils = h_max
    # h_stator_air = h_max
    # h_magnets = h_max
    # h_stator_iron = h_max
    # h_rotor_iron = h_max
    # h_shaft_iron = h_max

    h_air_gap = r7-r2 #2.0*(r6-r4) #0.05*h_max
    h_air_magnets = 1.5*h_air_gap #h_max
    h_coils = 4*h_air_gap #h_max
    h_stator_air = h_max
    h_magnets = 3*h_air_gap #h_max
    h_stator_iron = h_max
    h_rotor_iron = h_max
    h_shaft_iron = h_max

    pnt_origin = Pnt(0, 0, 0)
    pnt_r2_augmented_left   = Pnt(r2_augmented, 0, 0)
    pnt_r2_augmented_center = Pnt(r2_augmented*sqrt(2+sqrt(2))/2, r2_augmented*sqrt(2-sqrt(2))/2, 0)
    pnt_r2_augmented_right  = Pnt(r2_augmented*sqrt(2)/2, r2_augmented*sqrt(2)/2, 0)
    pnt_r4_left   = Pnt(r4, 0, 0)
    pnt_r4_center = Pnt(r4*sqrt(2+sqrt(2))/2, r4*sqrt(2-sqrt(2))/2, 0)
    pnt_r4_right  = Pnt(r4*sqrt(2)/2, r4*sqrt(2)/2, 0)
    pnt_r6_left   = Pnt(r6, 0, 0)
    pnt_r6_center = Pnt(r6*sqrt(2+sqrt(2))/2, r6*sqrt(2-sqrt(2))/2, 0)
    pnt_r6_right  = Pnt(r6*sqrt(2)/2, r6*sqrt(2)/2, 0)
    pnt_r1_left   = Pnt(r1, 0, 0)
    pnt_r1_center = Pnt(r1*sqrt(2+sqrt(2))/2, r1*sqrt(2-sqrt(2))/2, 0)
    pnt_r1_right  = Pnt(r1*sqrt(2)/2, r1*sqrt(2)/2, 0)
    pnt_r2_left   = Pnt(r2, 0, 0)
    pnt_r2_center = Pnt(r2*sqrt(2+sqrt(2))/2, r2*sqrt(2-sqrt(2))/2, 0)
    pnt_r2_right  = Pnt(r2*sqrt(2)/2, r2*sqrt(2)/2, 0)
    pnt_r7_left   = Pnt(r7, 0, 0)
    pnt_r7_center = Pnt(r7*sqrt(2+sqrt(2))/2, r7*sqrt(2-sqrt(2))/2, 0)
    pnt_r7_right  = Pnt(r7*sqrt(2)/2, r7*sqrt(2)/2, 0)
    pnt_r8_left   = Pnt(r8, 0, 0)
    pnt_r8_center = Pnt(r8*sqrt(2+sqrt(2))/2, r8*sqrt(2-sqrt(2))/2, 0)
    pnt_r8_right  = Pnt(r8*sqrt(2)/2, r8*sqrt(2)/2, 0)

    shaft_iron_left  = Segment(pnt_origin, pnt_r1_left)
    shaft_iron_right = Segment(pnt_r1_right, pnt_origin)

    rotor_left  = Segment(pnt_r1_left,  pnt_r2_left)
    rotor_right = Segment(pnt_r2_right, pnt_r1_right)

    if separated_mesh:
        air_gap_left  = Segment(pnt_r2_augmented_left,  pnt_r7_left)
        air_gap_right = Segment(pnt_r7_right, pnt_r2_augmented_right)
    else:
        air_gap_left  = Segment(pnt_r2_left,  pnt_r7_left)
        air_gap_right = Segment(pnt_r7_right, pnt_r2_right)

    stator_iron_left  = Segment(pnt_r7_left,  pnt_r8_left)
    stator_iron_right = Segment(pnt_r8_right, pnt_r7_right)

    sliding_inner = ArcOfCircle(pnt_r2_augmented_left, pnt_r2_augmented_center, pnt_r2_augmented_right)
    rotor_inner   = ArcOfCircle(pnt_r1_left, pnt_r1_center, pnt_r1_right)
    rotor_outer   = ArcOfCircle(pnt_r2_left, pnt_r2_center, pnt_r2_right)
    stator_inner  = ArcOfCircle(pnt_r7_left, pnt_r7_center, pnt_r7_right)
    stator_outer  = ArcOfCircle(pnt_r8_left, pnt_r8_center, pnt_r8_right)

    rotor_iron_w = Wire([rotor_left, rotor_outer, rotor_right, rotor_inner])
    rotor_iron = Face(rotor_iron_w)

    if separated_mesh:
        air_gap_w = Wire([air_gap_left, stator_inner, air_gap_right, sliding_inner])
    else:
        air_gap_w = Wire([air_gap_left, stator_inner, air_gap_right, rotor_outer])

    air_gap = Face(air_gap_w)

    stator_iron_w = Wire([stator_iron_left, stator_outer, stator_iron_right, stator_inner])
    stator_iron = Face(stator_iron_w)

    shaft_iron_w = Wire([shaft_iron_left, rotor_inner, shaft_iron_right])
    shaft_iron = Face(shaft_iron_w)

    stator_iron.edges[1].name = "stator_outer"

    shaft_iron.edges[0].name = "left"
    rotor_iron.edges[0].name = "left"
    air_gap.edges[0].name = "left"
    stator_iron.edges[0].name = "left"

    shaft_iron.edges[2].name = "right"
    rotor_iron.edges[2].name = "right"
    air_gap.edges[2].name = "right"
    stator_iron.edges[2].name = "right"

    rot = Rotation(Axis((0,0,0), Z), 45)
    shaft_iron.edges[0].Identify(shaft_iron.edges[2], "periodic", IdentificationType.PERIODIC, rot)
    rotor_iron.edges[0].Identify(rotor_iron.edges[2], "periodic", IdentificationType.PERIODIC, rot)
    air_gap.edges[0].Identify(air_gap.edges[2], "periodic", IdentificationType.PERIODIC, rot)
    stator_iron.edges[0].Identify(stator_iron.edges[2], "periodic", IdentificationType.PERIODIC, rot)

    string_coils = ""
    domain_name_coil = {0: "coil1", 1: "coil2", 2: "coil3", 3:"coil4", 4:"coil5",
                        5:"coil6", 6:"coil7", 7:"coil8", 8: "coil9", 9: "coil10", 10: "coil11", 11:"coil12",
                        12:"coil13",13: "coil14", 14: "coil15", 15: "coil16", 16:"coil17", 17:"coil18",
                        18: "coil19", 19: "coil20", 20: "coil21", 21:"coil22", 22:"coil23",
                        23: "coil24", 24: "coil25", 25: "coil26", 26:"coil27", 27:"coil28",
                        28: "coil29", 29: "coil30", 30: "coil31", 31:"coil32", 32:"coil33",
                        33: "coil34", 34: "coil35", 35: "coil36", 36:"coil37", 37:"coil38",
                        38: "coil39", 39: "coil40", 40: "coil41", 41:"coil42", 42:"coil43",
                        43: "coil44", 44: "coil45", 45: "coil46", 46:"coil47", 47:"coil48"}

    for k in range(6):#48
        (stator_coil,stator_air) = drawStatorNut(k)

        stator_coil.faces.name = domain_name_coil[k]
        stator_air.faces.name = "air"

        stator_iron -= stator_coil
        stator_iron -= stator_air

        stator_coil.faces.maxh = h_coils
        domains.append(stator_coil)
        domains.append(stator_air)
        string_coils += domain_name_coil[k] + "|"

    domains_magnets = [];
    domains_air_magnets = [];

    domain_name_magnet1 = {0: "magnet1", 1: "magnet3", 2: "magnet5", 3:"magnet7", 4:"magnet9",
                        5:"magnet11", 6:"magnet13", 7:"magnet15"}
    domain_name_magnet2 = {0: "magnet2", 1: "magnet4", 2: "magnet6", 3:"magnet8", 4:"magnet10",
                        5:"magnet12", 6:"magnet14", 7:"magnet16"}

    for k in range(1):#8
        (magnet1,air_magnet1_1,air_magnet1_2) = drawMagnet1(k)
        (magnet2,air_magnet2_1,air_magnet2_2) = drawMagnet2(k)

        magnet1.faces.name = domain_name_magnet1[k]
        magnet1.faces.maxh = h_magnets
        magnet1.edges[0].name = "magnets_interface"
        magnet1.edges[1].name = "magnets_interface"
        magnet1.edges[2].name = "magnets_interface"
        magnet1.edges[3].name = "magnets_interface"

        magnet2.faces.name = domain_name_magnet2[k]
        magnet2.faces.maxh = h_magnets
        magnet2.edges[0].name = "magnets_interface"
        magnet2.edges[1].name = "magnets_interface"
        magnet2.edges[2].name = "magnets_interface"
        magnet2.edges[3].name = "magnets_interface"

        air_magnet1_1.faces.name = "rotor_air"
        air_magnet1_2.faces.name = "rotor_air"
        air_magnet2_1.faces.name = "rotor_air"
        air_magnet2_2.faces.name = "rotor_air"

        air_magnet1_1.faces.maxh = h_air_magnets
        air_magnet1_2.faces.maxh = h_air_magnets
        air_magnet2_1.faces.maxh = h_air_magnets
        air_magnet2_2.faces.maxh = h_air_magnets

        rotor_iron -= magnet1
        rotor_iron -= air_magnet1_1;
        rotor_iron -= air_magnet1_2;
        rotor_iron -= magnet2
        rotor_iron -= air_magnet2_1;
        rotor_iron -= air_magnet2_2;

        domains.append(magnet1)
        domains.append(magnet2)
        domains.append(air_magnet1_1)
        domains.append(air_magnet1_2)
        domains.append(air_magnet2_1)
        domains.append(air_magnet2_2)

    stator_iron.faces.name = "stator_iron"
    stator_iron.faces.maxh = h_stator_iron

    air_gap.faces.maxh = h_air_gap
    air_gap.faces.name = "air_gap"
    air_gap.edges[3].maxh = h_air_gap
    air_gap.edges[3].name = "airgap_inner"

    shaft_iron.faces.name = "shaft_iron"
    shaft_iron.faces.maxh = h_shaft_iron

    rotor_iron.faces.name = "rotor_iron"
    rotor_iron.faces.maxh = h_rotor_iron
    rotor_iron.edges[2].maxh = h_air_gap
    rotor_iron.edges[2].name = "rotor_outer"

    domains.append(rotor_iron)
    domains.append(air_gap)
    domains.append(stator_iron)
    domains.append(shaft_iron)

    geo = Glue(domains)
    geoOCC = OCCGeometry(geo, dim=2)

    mesh = ngsolve.Mesh(geoOCC.GenerateMesh())
    mesh.ngmesh.Save("Motor_Bosch_2d_antiperiodic.vol")
else:
    mesh = ngsolve.Mesh("Motor_Bosch_2d_antiperiodic.vol")
    #inner radius rotor
    r1 = 26.5*10**(-3);
    #outer radius rotor
    r2_original = 78.63225*10**(-3);
    r2 =  78.62725*10**(-3);
    #sliding mesh rotor
    r4_reduced = 78.80*10**(-3);
    r4 = 78.8354999*10**(-3);
    #sliding mesh stator
    r6 = 79.03874999*10**(-3);
    #inner radius stator
    r7 = 79.242*10**(-3);
    #outer radius stator
    r8 = 116*10**(-3)

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

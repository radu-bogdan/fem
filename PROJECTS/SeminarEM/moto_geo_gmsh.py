import gmsh
import numpy as np

orign = (0,0);
#inner radius rotor
r1 = 26.5*10**(-3);
#outer radius rotor
r2 = 78.63225*10**(-3);
#sliding mesh rotor
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

def drawMagnet1(k):
    m1xnew = m1[0]*np.cos(k*np.pi/4) -m1[1]*np.sin(k*np.pi/4)
    m1ynew = m1[0]*np.sin(k*np.pi/4) +m1[1]*np.cos(k*np.pi/4)
    m1new = gmsh.model.occ.addPoint(m1xnew,m1ynew,0)

    m2xnew = m2[0]*np.cos(k*np.pi/4) -m2[1]*np.sin(k*np.pi/4)
    m2ynew = m2[0]*np.sin(k*np.pi/4) +m2[1]*np.cos(k*np.pi/4)
    m2new = gmsh.model.occ.addPoint(m2xnew,m2ynew,0)

    m3xnew = m3[0]*np.cos(k*np.pi/4) -m3[1]*np.sin(k*np.pi/4)
    m3ynew = m3[0]*np.sin(k*np.pi/4) +m3[1]*np.cos(k*np.pi/4)
    m3new = gmsh.model.occ.addPoint(m3xnew,m3ynew,0)

    m4xnew = m4[0]*np.cos(k*np.pi/4) -m4[1]*np.sin(k*np.pi/4)
    m4ynew = m4[0]*np.sin(k*np.pi/4) +m4[1]*np.cos(k*np.pi/4)
    m4new = gmsh.model.occ.addPoint(m4xnew,m4ynew,0)

    a5xnew = a5[0]*np.cos(k*np.pi/4) -a5[1]*np.sin(k*np.pi/4)
    a5ynew = a5[0]*np.sin(k*np.pi/4) +a5[1]*np.cos(k*np.pi/4)
    a5new = gmsh.model.occ.addPoint(a5xnew,a5ynew,0)

    a6xnew = a6[0]*np.cos(k*np.pi/4) -a6[1]*np.sin(k*np.pi/4)
    a6ynew = a6[0]*np.sin(k*np.pi/4) +a6[1]*np.cos(k*np.pi/4)
    a6new = gmsh.model.occ.addPoint(a6xnew,a6ynew,0)

    a7xnew = a7[0]*np.cos(k*np.pi/4) -a7[1]*np.sin(k*np.pi/4)
    a7ynew = a7[0]*np.sin(k*np.pi/4) +a7[1]*np.cos(k*np.pi/4)
    a7new = gmsh.model.occ.addPoint(a7xnew,a7ynew,0)

    a8xnew = a8[0]*np.cos(k*np.pi/4) -a8[1]*np.sin(k*np.pi/4)
    a8ynew = a8[0]*np.sin(k*np.pi/4) +a8[1]*np.cos(k*np.pi/4)
    a8new = gmsh.model.occ.addPoint(a8xnew,a8ynew,0)
    
    seg1 = gmsh.model.occ.addLine(m1new, m2new)
    seg2 = gmsh.model.occ.addLine(m2new, m3new)
    seg3 = gmsh.model.occ.addLine(m3new, m4new)
    seg4 = gmsh.model.occ.addLine(m4new, m1new)
    
    magnet1 = gmsh.model.occ.addCurveLoop([seg1,seg2,seg3,seg4])
    magnet1_surface = gmsh.model.occ.addPlaneSurface([magnet1])
    
    phd = gmsh.model.addPhysicalGroup(dim = 2, tags = [magnet1], name = 'magnet' + str(k))
    gmsh.model.setPhysicalName(dim = 2, tag = phd, name = 'magnet' + str(k))

    
    air_seg1 = gmsh.model.occ.addLine(m1new,a5new)
    air_seg2 = gmsh.model.occ.addLine(a5new,a6new)
    air_seg3 = gmsh.model.occ.addLine(a6new,m2new)
    air_seg4 = gmsh.model.occ.addLine(m2new,m1new)
    
    air_magnet1_1 = gmsh.model.occ.addCurveLoop([air_seg1,air_seg2,air_seg3,air_seg4])
    air_magnet1_1_surface = gmsh.model.occ.addPlaneSurface([air_magnet1_1])
    
    air_seg5 = gmsh.model.occ.addLine(m4new,m3new)
    air_seg6 = gmsh.model.occ.addLine(m3new,a7new)
    air_seg7 = gmsh.model.occ.addLine(a7new,a8new)
    air_seg8 = gmsh.model.occ.addLine(a8new,m4new)
    
    air_magnet1_2 = gmsh.model.occ.addCurveLoop([air_seg5,air_seg6,air_seg7,air_seg8])
    air_magnet1_2_surface = gmsh.model.occ.addPlaneSurface([air_magnet1_2])
    
    
def drawMagnet2(k):
    m5xnew = m5[0]*np.cos(k*np.pi/4) -m5[1]*np.sin(k*np.pi/4)
    m5ynew = m5[0]*np.sin(k*np.pi/4) +m5[1]*np.cos(k*np.pi/4)
    m5new = gmsh.model.occ.addPoint(m5xnew ,m5ynew ,0 ,meshSize = 0)

    m6xnew = m6[0]*np.cos(k*np.pi/4) -m6[1]*np.sin(k*np.pi/4)
    m6ynew = m6[0]*np.sin(k*np.pi/4) +m6[1]*np.cos(k*np.pi/4)
    m6new = gmsh.model.occ.addPoint(m6xnew,m6ynew,0)

    m7xnew = m7[0]*np.cos(k*np.pi/4) -m7[1]*np.sin(k*np.pi/4)
    m7ynew = m7[0]*np.sin(k*np.pi/4) +m7[1]*np.cos(k*np.pi/4)
    m7new = gmsh.model.occ.addPoint(m7xnew,m7ynew,0)

    m8xnew = m8[0]*np.cos(k*np.pi/4) -m8[1]*np.sin(k*np.pi/4)
    m8ynew = m8[0]*np.sin(k*np.pi/4) +m8[1]*np.cos(k*np.pi/4)
    m8new = gmsh.model.occ.addPoint(m8xnew,m8ynew,0)

    a1xnew = a1[0]*np.cos(k*np.pi/4) -a1[1]*np.sin(k*np.pi/4)
    a1ynew = a1[0]*np.sin(k*np.pi/4) +a1[1]*np.cos(k*np.pi/4)
    a1new = gmsh.model.occ.addPoint(a1xnew,a1ynew,0)

    a2xnew = a2[0]*np.cos(k*np.pi/4) -a2[1]*np.sin(k*np.pi/4)
    a2ynew = a2[0]*np.sin(k*np.pi/4) +a2[1]*np.cos(k*np.pi/4)
    a2new = gmsh.model.occ.addPoint(a2xnew,a2ynew,0)

    a3xnew = a3[0]*np.cos(k*np.pi/4) -a3[1]*np.sin(k*np.pi/4)
    a3ynew = a3[0]*np.sin(k*np.pi/4) +a3[1]*np.cos(k*np.pi/4)
    a3new = gmsh.model.occ.addPoint(a3xnew,a3ynew,0)

    a4xnew = a4[0]*np.cos(k*np.pi/4) -a4[1]*np.sin(k*np.pi/4)
    a4ynew = a4[0]*np.sin(k*np.pi/4) +a4[1]*np.cos(k*np.pi/4)
    a4new = gmsh.model.occ.addPoint(a4xnew,a4ynew,0)
    
    seg1 = gmsh.model.occ.addLine(m5new, m6new)
    seg2 = gmsh.model.occ.addLine(m6new, m7new)
    seg3 = gmsh.model.occ.addLine(m7new, m8new)
    seg4 = gmsh.model.occ.addLine(m8new, m5new)
    
    magnet2 = gmsh.model.occ.addCurveLoop([seg1,seg2,seg3,seg4])
    magnet2_surface = gmsh.model.occ.addPlaneSurface([magnet2])
    
    air_seg1 = gmsh.model.occ.addLine(m5new,a3new)
    air_seg2 = gmsh.model.occ.addLine(a3new,a4new)
    air_seg3 = gmsh.model.occ.addLine(a4new,m6new)
    air_seg4 = gmsh.model.occ.addLine(m6new,m5new)
    
    air_magnet2_1 = gmsh.model.occ.addCurveLoop([air_seg1,air_seg2,air_seg3,air_seg4])
    # air_magnet2_1_surface = gmsh.model.occ.addPlaneSurface([air_magnet2_1])
    
    air_seg5 = gmsh.model.occ.addLine(m8new,m7new)
    air_seg6 = gmsh.model.occ.addLine(m7new,a2new)
    air_seg7 = gmsh.model.occ.addLine(a2new,a1new)
    air_seg8 = gmsh.model.occ.addLine(a1new,m8new)
    
    air_magnet2_2 = gmsh.model.occ.addCurveLoop([air_seg5,air_seg6,air_seg7,air_seg8])
    
    air_magnet2_1_surface = gmsh.model.occ.addPlaneSurface([air_magnet2_2,air_magnet2_1])
        
def drawStatorNut(k):
    s1xnew = s1[0]*np.cos(k*np.pi/24) -s1[1]*np.sin(k*np.pi/24)
    s1ynew = s1[0]*np.sin(k*np.pi/24) +s1[1]*np.cos(k*np.pi/24)
    s1new = gmsh.model.occ.addPoint(s1xnew,s1ynew,0)

    s2xnew = s2[0]*np.cos(k*np.pi/24) -s2[1]*np.sin(k*np.pi/24)
    s2ynew = s2[0]*np.sin(k*np.pi/24) +s2[1]*np.cos(k*np.pi/24)
    s2new = gmsh.model.occ.addPoint(s2xnew,s2ynew,0)

    s3xnew = s3[0]*np.cos(k*np.pi/24) -s3[1]*np.sin(k*np.pi/24)
    s3ynew = s3[0]*np.sin(k*np.pi/24) +s3[1]*np.cos(k*np.pi/24)
    s3new = gmsh.model.occ.addPoint(s3xnew,s3ynew,0)

    s4xnew = s4[0]*np.cos(k*np.pi/24) -s4[1]*np.sin(k*np.pi/24)
    s4ynew = s4[0]*np.sin(k*np.pi/24) +s4[1]*np.cos(k*np.pi/24)
    s4new = gmsh.model.occ.addPoint(s4xnew,s4ynew,0)

    s5xnew = s5[0]*np.cos(k*np.pi/24) -s5[1]*np.sin(k*np.pi/24)
    s5ynew = s5[0]*np.sin(k*np.pi/24) +s5[1]*np.cos(k*np.pi/24)
    s5new = gmsh.model.occ.addPoint(s5xnew,s5ynew,0)

    s6xnew = s6[0]*np.cos(k*np.pi/24) -s6[1]*np.sin(k*np.pi/24)
    s6ynew = s6[0]*np.sin(k*np.pi/24) +s6[1]*np.cos(k*np.pi/24)
    s6new = gmsh.model.occ.addPoint(s6xnew,s6ynew,0)

    s7xnew = s7[0]*np.cos(k*np.pi/24) -s7[1]*np.sin(k*np.pi/24)
    s7ynew = s7[0]*np.sin(k*np.pi/24) +s7[1]*np.cos(k*np.pi/24)
    s7new = gmsh.model.occ.addPoint(s7xnew,s7ynew,0)

    s8xnew = s8[0]*np.cos(k*np.pi/24) -s8[1]*np.sin(k*np.pi/24)
    s8ynew = s8[0]*np.sin(k*np.pi/24) +s8[1]*np.cos(k*np.pi/24)
    s8new = gmsh.model.occ.addPoint(s8xnew,s8ynew,0)

    #Draw stator coil
    seg1 = gmsh.model.occ.addLine(s2new,s3new)
    seg2 = gmsh.model.occ.addLine(s3new,s4new)
    seg3 = gmsh.model.occ.addLine(s4new,s5new)
    seg4 = gmsh.model.occ.addLine(s5new,s6new)
    seg5 = gmsh.model.occ.addLine(s6new,s7new)
    seg6 = gmsh.model.occ.addLine(s7new,s2new)
    stator_coil = gmsh.model.occ.addCurveLoop([seg1,seg2,seg3,seg4,seg5,seg6])
    
    #Draw air nut in the stator
    air_seg1 = gmsh.model.occ.addLine(s1new,s2new)
    air_seg2 = gmsh.model.occ.addLine(s2new,s7new)
    air_seg3 = gmsh.model.occ.addLine(s7new,s8new)
    air_seg4 = gmsh.model.occ.addLine(s8new,s1new)
    stator_air = gmsh.model.occ.addCurveLoop([air_seg1,air_seg2,air_seg3,air_seg4])
    
    # gmsh.model.occ.fuse([(3, stator_air)], [(3, b)])
    
    # stator_air = stator_air-(stator_air*air_gap_stator)

gmsh.initialize()

rotor_inner  = gmsh.model.occ.addCircle(0,0,0,r1)
rotor_outer  = gmsh.model.occ.addCircle(0,0,0,r2)
sliding_inner  = gmsh.model.occ.addCircle(0,0,0,r4)
sliding_outer  = gmsh.model.occ.addCircle(0,0,0,r6)
stator_inner = gmsh.model.occ.addCircle(0,0,0,r7)
stator_outer = gmsh.model.occ.addCircle(0,0,0,r8)

# gmsh.model.occ.add

for i in range(8):
    drawMagnet1(i)
    drawMagnet2(i)  
    
for i in range(48):
    drawStatorNut(i)

air_gap_stator = stator_inner - sliding_outer

# gmsh.model.geo.synchronize()

gmsh.model.occ.synchronize()
# gmsh.model.occ.healShapes()

# gmsh.write("whateva.msh")

gmsh.fltk.run()
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.finalize()


# gmsh.model.occ.fuse
    
from ngsolve import *
from netgen.geom2d import SplineGeometry
import numpy as np
# import multistruct as multi
# import netgen.gui

rai_d=1e-3
#inner radius rotor
rdi = 26.5*10**(-3)
#outer radius rotor
rdo = 78.63225*10**(-3)
#interface
ragi= 78.802166667*10**(-3)
rago = 78.972083327*10**(-3)
#inner radius stator
rsi = 79.142*10**(-3)
#outer radius stator
rso = 116*10**(-3)

pai_d=[(rai_d,0),(rai_d,rai_d*(sqrt(2)-1)),(rai_d/sqrt(2),rai_d/sqrt(2))]
pdi=[(rdi,0),(rdi,rdi*(sqrt(2)-1)),(rdi/sqrt(2),rdi/sqrt(2))]
pdo=[(rdo,0),(rdo,rdo*(sqrt(2)-1)),(rdo/sqrt(2),rdo/sqrt(2))]
pagi=[(ragi,0),(ragi,ragi*(sqrt(2)-1)),(ragi/sqrt(2),ragi/sqrt(2))]
pago=[(rago,0),(rago,rago*(sqrt(2)-1)),(rago/sqrt(2),rago/sqrt(2))]
psi=[(rsi,0),(rsi,rsi*(sqrt(2)-1)),(rsi/sqrt(2),rsi/sqrt(2))]
pso=[(rso,0),(rso,rso*(sqrt(2)-1)),(rso/sqrt(2),rso/sqrt(2))]
ps0=(79.142124714*1e-3,1.9781518706*1e-3)
ps1=(57.360697664698975017927295994014*1e-3,54.563168460862740971606399398297*1e-3)
# pao_d=[(rao_d,0),(rao_d,rao_d),(0,rao_d)]


# domainnum=[4]*8+[5]*8+[6]*8+[7]*8+[8]*8+[9]*8   #1 pole pairs
# domainnum=([4]*4+[5]*4+[6]*4+[7]*4+[8]*4+[9]*4)*2    #2 pole pairs
domainnum=([4]*2+[5]*2+[6]*2+[7]*2+[8]*2+[9]*2)*4    #4 pole pairs

nMagnets=1 #half!!

geo = SplineGeometry()

Pai_d=[ geo.AppendPoint(*pnt) for pnt in pai_d ]
Pdi=[ geo.AppendPoint(*pnt) for pnt in pdi ]
Pdo=[ geo.AppendPoint(*pnt) for pnt in pdo ]
Pagi=[ geo.AppendPoint(*pnt) for pnt in pagi ]
Pago=[ geo.AppendPoint(*pnt) for pnt in pago ]
Psi=[ geo.AppendPoint(*pnt) for pnt in psi ]
Pso=[ geo.AppendPoint(*pnt) for pnt in pso ]
# Pao_d=[ geo.AppendPoint(*pnt) for pnt in pao_d ]
Ps0=geo.AppendPoint(*ps0)
Ps1=geo.AppendPoint(*ps1)
# geo.AddCircle((0,0),rdi,leftdomain=10,rightdomain=2)
# geo.AddCircle((0,0),rdo,leftdomain=2,rightdomain=10)
# geo.AddCircle((0,0),ragi,leftdomain=10,rightdomain=0,bc="interin")
# geo.AddCircle((0,0),rago,leftdomain=0,rightdomain=1,bc="interout")
# geo.AddCircle((0,0),rso,leftdomain=3,rightdomain=0,bc="diri")

### down right lines, y=0
y01=geo.Append(['line',Pai_d[0],Pdi[0]],leftdomain=1,rightdomain=0)
y02=geo.Append(['line',Pdi[0],Pdo[0]],leftdomain=2,rightdomain=0)
y03=geo.Append(['line',Pdo[0],Pagi[0]],leftdomain=1,rightdomain=0)
y04i=geo.Append(['line',Pagi[0],Pago[0]],leftdomain=10,rightdomain=0)
y04=geo.Append(['line',Pago[0],Psi[0]],leftdomain=1,rightdomain=0)
y05=geo.Append(['line',Psi[0],Pso[0]],leftdomain=3,rightdomain=0)
# y06=geo.Append(['line',Pso[0],Pao_d[0]],leftdomain=1,rightdomain=0)

### upper left lines, x=0
geo.Append(['line',Pai_d[2],Pdi[2]],leftdomain=0,rightdomain=1,copy=y01)
geo.Append(['line',Pdi[2],Pdo[2]],leftdomain=0,rightdomain=2,copy=y02)
geo.Append(['line',Pdo[2],Pagi[2]],leftdomain=0,rightdomain=1,copy=y03)
geo.Append(['line',Pagi[2],Pago[2]],leftdomain=0,rightdomain=10,copy=y04i)
geo.Append(['line',Pago[2],Psi[2]],leftdomain=0,rightdomain=1,copy=y04)
geo.Append(['line',Psi[2],Pso[2]],leftdomain=0,rightdomain=3,copy=y05)
# geo.Append(['line',Pso[2],Pao_d[2]],leftdomain=0,rightdomain=1,copy=y06)

geo.Append(['spline3',Pai_d[0],Pai_d[1],Pai_d[2]],leftdomain=0,rightdomain=1,bc="diri_in")
geo.Append(['spline3',Pdi[0],Pdi[1],Pdi[2]],leftdomain=1,rightdomain=2,bc="diri_el")
geo.Append(['spline3',Pdo[0],Pdo[1],Pdo[2]],leftdomain=2,rightdomain=1)
geo.Append(['spline3',Pagi[0],Pagi[1],Pagi[2]],leftdomain=1,rightdomain=10)
geo.Append(['spline3',Pago[0],Pago[1],Pago[2]],leftdomain=10,rightdomain=1,bc="inter")
# geo.Append(['spline3',Psi[0],Psi[1],Psi[2]],leftdomain=1,rightdomain=3)
geo.Append(['spline3',Pso[0],Pso[1],Pso[2]],leftdomain=3,rightdomain=0,bc="diri_out")
# geo.Append(['spline3',Pao_d[0],Pao_d[1],Pao_d[2]],leftdomain=1,rightdomain=0,bc="diri_out")

#Points for Stator Nut and air in the stator
s1 = (79.04329892000*10**(-3),3.9538335974*10**(-3))
s2 = (80.143057128*10**(-3),4.0037794254*10**(-3))
s3 = (80.387321219*10**(-3),2.965459706*10**(-3))
s4 = (98.78501315600001*10**(-3),3.9007973292*10**(-3))
s5 = (98.44904989600001*10**(-3),9.026606148400001*10**(-3))
s6 = (80.086666706*10**(-3),7.5525611543*10**(-3))
s7 = (79.980020247*10**(-3),6.4912415424*10**(-3))
s8 = (78.88229587*10**(-3),6.4102654448*10**(-3))
s9 = (0.0786240034,0.0103510464)

pStatorRef=[s1,s2,s3,s4,s5,s6,s7,s8,s9]

StatorRot=[0.9914448613738104111445575269285628712777382744481022714587746035 ,0.1305261922200515915484062278954890101937407048117322518906169483 ]
pStator=pStatorRef
p=[ geo.AppendPoint(*pnt) for pnt in pStator ]

for i in range(6):
    geo.Append(['line',p[0+9*i],p[1+9*i]],rightdomain=3,leftdomain=1)
    geo.Append(['line',p[1+9*i],p[2+9*i]],rightdomain=3,leftdomain=domainnum[i])
    geo.Append(['line',p[2+9*i],p[3+9*i]],rightdomain=3,leftdomain=domainnum[i])
    geo.Append(['line',p[3+9*i],p[4+9*i]],rightdomain=3,leftdomain=domainnum[i])
    geo.Append(['line',p[4+9*i],p[5+9*i]],rightdomain=3,leftdomain=domainnum[i])
    geo.Append(['line',p[5+9*i],p[6+9*i]],rightdomain=3,leftdomain=domainnum[i])
    geo.Append(['line',p[6+9*i],p[1+9*i]],rightdomain=1,leftdomain=domainnum[i])
    geo.Append(['line',p[6+9*i],p[7+9*i]],rightdomain=3,leftdomain=1)

    if i==0:
        geo.Append(['spline3',Psi[0],Ps0,p[0]],leftdomain=1,rightdomain=3)
        pStator=[(StatorRot[0]*pnt[0]-StatorRot[1]*pnt[1],StatorRot[1]*pnt[0]+StatorRot[0]*pnt[1]) for pnt in pStator]
        p+= [ geo.AppendPoint(*pnt) for pnt in pStator ]
        geo.Append(['spline3',p[7+9*i],p[8+9*i],p[0+9*(i+1)]],leftdomain=1,rightdomain=3)
    elif i==5:
        geo.Append(['spline3',p[7+9*i],Ps1,Psi[2]],leftdomain=1,rightdomain=3)
    else:
        pStator=[(StatorRot[0]*pnt[0]-StatorRot[1]*pnt[1],StatorRot[1]*pnt[0]+StatorRot[0]*pnt[1]) for pnt in pStator]
        p+= [ geo.AppendPoint(*pnt) for pnt in pStator ]
        geo.Append(['spline3',p[7+9*i],p[8+9*i],p[0+9*(i+1)]],leftdomain=1,rightdomain=3)


#Points for magnet1 and air around magnet1
m1 = (69.23112999*10**(-3),7.535512*10**(-3))
m2 = (74.828958945*10**(-3),10.830092744*10**(-3))
m3 = (66.13621099700001*10**(-3),25.599935335*10**(-3))
m4 = (60.53713*10**(-3),22.30748*10**(-3))
a5 = (69.75636*10**(-3),5.749913*10**(-3))
a6 = (75.06735*10**(-3),3.810523*10**(-3))
a7 = (65.3506200*10**(-3),26.51379*10**(-3))
a7 = (65.6868747*10**(-3),26.3184618*10**(-3))
a8 = (59.942145092*10**(-3),24.083661604*10**(-3))
#Points for magnet2 and air around magnet2
m5 = (58.579985516*10**(-3), 27.032444757*10**(-3))
m6 = (64.867251151*10**(-3),28.663475405*10**(-3))
m7 = (60.570096319*10**(-3),45.254032279*10**(-3))
m8 = (54.282213127*10**(-3),43.625389857*10**(-3))
a1 = (53.39099766*10**(-3),45.259392713*10**(-3))
a2 = (55.775078884*10**(-3),50.386185578*10**(-3))
a3 = (59.41521771*10**(-3),25.355776837*10**(-3))
a4 = (65.12210917100001*10**(-3),27.707477175*10**(-3))

M1ref=[m1,m2,m3,m4]
A1ref=[a5,a6,a7,a8]
M2ref=[m5,m6,m7,m8]
A2ref=[a3,a4,a2,a1]


MagnetRot=[np.cos(2*pi/(nMagnets)) ,np.sin(2*pi/(nMagnets)) ]
M1=M1ref
A1=A1ref
M2=M2ref
A2=A2ref

for i in range(nMagnets):
    pM1=[ geo.AppendPoint(*pnt) for pnt in M1 ]
    pM2=[ geo.AppendPoint(*pnt) for pnt in M2 ]
    pA1=[ geo.AppendPoint(*pnt) for pnt in A1 ]
    pA2=[ geo.AppendPoint(*pnt) for pnt in A2 ]
    
    geo.Append(['line',pM1[0],pM1[1]],leftdomain=11+2*i,rightdomain=10)
    geo.Append(['line',pM1[1],pM1[2]],leftdomain=11+2*i,rightdomain=2)
    geo.Append(['line',pM1[2],pM1[3]],leftdomain=11+2*i,rightdomain=10)
    geo.Append(['line',pM1[3],pM1[0]],leftdomain=11+2*i,rightdomain=2)
    geo.Append(['line',pM1[0],pA1[0]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pA1[0],pA1[1]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pA1[1],pM1[1]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pM1[2],pA1[2]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pA1[2],pA1[3]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pA1[3],pM1[3]],leftdomain=10,rightdomain=2)
    
    geo.Append(['line',pM2[0],pM2[1]],leftdomain=12+2*i,rightdomain=10)
    geo.Append(['line',pM2[1],pM2[2]],leftdomain=12+2*i,rightdomain=2)
    geo.Append(['line',pM2[2],pM2[3]],leftdomain=12+2*i,rightdomain=10)
    geo.Append(['line',pM2[3],pM2[0]],leftdomain=12+2*i,rightdomain=2)
    geo.Append(['line',pM2[0],pA2[0]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pA2[0],pA2[1]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pA2[1],pM2[1]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pM2[2],pA2[2]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pA2[2],pA2[3]],leftdomain=10,rightdomain=2)
    geo.Append(['line',pA2[3],pM2[3]],leftdomain=10,rightdomain=2)    
 
    M1=[(MagnetRot[0]*pnt[0]-MagnetRot[1]*pnt[1],MagnetRot[1]*pnt[0]+MagnetRot[0]*pnt[1]) for pnt in M1]
    M2=[(MagnetRot[0]*pnt[0]-MagnetRot[1]*pnt[1],MagnetRot[1]*pnt[0]+MagnetRot[0]*pnt[1]) for pnt in M2]
    A1=[(MagnetRot[0]*pnt[0]-MagnetRot[1]*pnt[1],MagnetRot[1]*pnt[0]+MagnetRot[0]*pnt[1]) for pnt in A1]
    A2=[(MagnetRot[0]*pnt[0]-MagnetRot[1]*pnt[1],MagnetRot[1]*pnt[0]+MagnetRot[0]*pnt[1]) for pnt in A2]


geo.SetMaterial (1, "air")
geo.SetMaterial (2, "design")
geo.SetMaterial (3, "stator")
geo.SetMaterial (4, "U")
geo.SetMaterial (5, "-V")
geo.SetMaterial (6, "W")
geo.SetMaterial (7, "-U")
geo.SetMaterial (8, "V")
geo.SetMaterial (9, "-W")
geo.SetMaterial (10, "airgap")
strmag=""
for i in range(2*nMagnets):
    geo.SetMaterial(11+i,"M"+str(i+1))
    if i>0:
        strmag+="|"
    strmag+="M"+str(i+1)


geo.SetDomainMaxH(2,2e-3)

maxh = 0.0005
mesh = Mesh(geo.GenerateMesh(maxh=maxh)) #convert to a ngsolve mesh

#####################################################################################################################
regions_1d = mesh.GetBoundaries()
regions_2d = mesh.GetMaterials()

regions_1d_np = np.zeros((len(regions_1d),), dtype=object)
regions_1d_np[:] = list(regions_1d)

regions_2d_np = np.zeros((len(regions_2d),), dtype=object)
regions_2d_np[:] = list(regions_2d)

# p = []
# for k in mesh.vertices:
#     p += [np.array(mesh[k].point)]
# p = np.array(p)

p = []
for i, el in enumerate(mesh.ngmesh.Points()):
    p += [np.array([el[0],el[1]])]
p = np.array(p)

t = []
for i, el in enumerate(mesh.ngmesh.Elements2D()):
    t += [np.array([el.points[0].nr,\
                    el.points[1].nr,\
                    el.points[2].nr,\
                    # el.points[3].nr,\
                    # el.points[4].nr,\
                    # el.points[5].nr,\
                    el.index])-1]
t = np.array(t)

e = []
for i, el in enumerate(mesh.ngmesh.Elements1D()):
    e += [np.array([el.points[0].nr,\
                    el.points[1].nr,\
                    # el.points[2].nr,\
                    el.index])-1]
e = np.array(e)

q = np.empty(0)
#####################################################################################################################

import sys
sys.path.insert(0,'../../') # adds parent directory

import pde
MESH = pde.mesh(p,e,t,q,regions_2d,regions_1d)

MESH.pdemesh2()


stop

npper1=[]
npper2=[]
npperel1=[]
npperel2=[]
rotor_points=[]
pnts=mesh.ngmesh.Points()
for i,p in enumerate(pnts):
    if sqrt(p[0]**2+p[1]**2)<rago*(1+1e-3):
        rotor_points.append(p)
    if p[1]<1e-6 and p[0]>1e-8:
        npper1.append(i+1)
        if abs(p[0]-(rdi+rdo)/2)<(rdo-rdi)/2*(1+1e-3):
            npperel1.append(i+1)
    if abs(p[0]-p[1])<1e-6 and p[1]>1e-8:
        npper2.append(i+1)
        if abs(sqrt(p[0]**2+p[1]**2)-(rdi+rdo)/2)<(rdo-rdi)/2*(1+1e-3):
            npperel2.append(i+1)
        
pper1=[(pnts[i][0],pnts[i][1]) for i in npper1]
pper2=[(pnts[i][0],pnts[i][1]) for i in npper2]
index1=sorted(range(len(npper1)), key=lambda k: pper1[k][0])
npper1=[npper1[index1[i]] for i in range(len(npper1))]
index2=sorted(range(len(npper2)), key=lambda k: pper2[k][1])
npper2=[npper2[index2[i]] for i in range(len(npper2))]
npper1=[i-1 for i in npper1]
npper2=[i-1 for i in npper2]

pperel1=[(pnts[i][0],pnts[i][1]) for i in npperel1]
pperel2=[(pnts[i][0],pnts[i][1]) for i in npperel2]
index1=sorted(range(len(npperel1)), key=lambda k: pperel1[k][0])
npperel1=[npperel1[index1[i]] for i in range(len(npperel1))]
index2=sorted(range(len(npperel2)), key=lambda k: pperel2[k][1])
npperel2=[npperel2[index2[i]] for i in range(len(npperel2))]
npperel1=[i-1 for i in npperel1]
npperel2=[i-1 for i in npperel2]




mesh.ngmesh.Save("PMS2d.vol")
Draw(mesh)
print(mesh.nv)
input("§")


# Mperp_mag1 =  (-0.507223091788922, 0.861814791678634) 
# Mperp_mag2 =  (-0.250741225095427, 0.968054150364350) 
# Mperp_mag3 =  (0.968055971101187, -0.250734195544481) 
# Mperp_mag4 =  (0.861818474866413, -0.507216833690415) 
# Mperp_mag5 =  (-0.861814791678634, -0.507223091788922)
# Mperp_mag6 =  (-0.968054150364350, -0.250741225095427) 
# Mperp_mag7 =  (0.250734195544481, 0.968055971101187) 
# Mperp_mag8 =  (0.507216833690415, 0.861818474866413) 
# Mperp_mag9 =  (0.507223091788922, -0.861814791678634) 
# Mperp_mag10 = (0.250741225095427, -0.968054150364350) 
# Mperp_mag11 = (-0.968055971101187, 0.250734195544481) 
# Mperp_mag12 = (-0.861818474866413, 0.507216833690415) 
# Mperp_mag13 = (0.861814791678634, 0.507223091788922) 
# Mperp_mag14 = (0.968054150364350, 0.250741225095427) 
# Mperp_mag15 = (-0.250734195544481, -0.968055971101187)
# Mperp_mag16 = (-0.507216833690415, -0.861818474866413)

# Mperp_all=[Mperp_mag1,Mperp_mag2,Mperp_mag3,Mperp_mag4,Mperp_mag5,Mperp_mag6,Mperp_mag7,Mperp_mag8,Mperp_mag9,Mperp_mag10,Mperp_mag11,Mperp_mag12,Mperp_mag13,Mperp_mag14,Mperp_mag15,Mperp_mag16]

# cfMperp=CoefficientFunction([(0,0)]*10+Mperp_all)

nu0=1e7/(4*pi)
num=nu0/1.05
BR = 1.05*1.158095238095238
lz=0.1795
T  = 2* 1.0*0.03
rotation_angle = np.pi
Vrot=rotation_angle/T
nCoils=6
areaCoils = Integrate(cf=1, mesh=mesh, order = 2, definedon = mesh.Materials("U|V|W|-U|-V|-W"))
areaOfOneCoil = areaCoils/ nCoils
Volall=Integrate(cf=1,mesh=mesh,order=2,definedon=mesh.Materials("design"))

offset = 0
pp  = 4
gamma_correction_model = -30.0
gamma = 40.0
gamma_correction_timestep = -1
phi0 = (gamma + gamma_correction_model + gamma_correction_timestep * pp) * pi/180.0

I=1555.63491861*2.75 / areaOfOneCoil/2
t=Parameter(0.0)
IU=sin(phi0+4*t)
IV=sin(phi0+4*t+2*pi/3)
IW=sin(phi0+4*t+4*pi/3)
I_d = {"default" : 0, "airin" : 0, "airout" : 0, "stator" : 0, "design" : 0, "airgap": 0,
        "U": I*IU, "V": I*IV, "W": I*IW, "-U": -I*IU, "-V": -I*IV, "-W": -I*IW}
cfI=CoefficientFunction([I_d[mat] for mat in mesh.GetMaterials()])
# cfNu=CoefficientFunction([nu0,0,0]+[nu0]*7+[num]*2)
# cfI=CoefficientFunction([0,0,0,I*IU,-I*IV,I*IW]+[0]*6)


draw_d = {"default" : 0, "airin" : 0, "airout" : 0, "stator" : 1, "design" : 0, "airgap": 0,
        "U": 0.75, "V": 0.75, "W": 0.75, "-U": 0.75, "-V": 0.75, "-W": 0.75}
cfDraw=CoefficientFunction([draw_d[mat] for mat in mesh.GetMaterials()])
matcolors = [1]+list(np.linspace(0.2,0.8,multi.nMagnets))+[0] ########### Draw-color/value of magnets
if not len(matcolors)==multi.M:
    matcolors = [1]+[0.4]*multi.nMagnets+[0]
L=L2(mesh,order=0)
cutratio=[]
for i in range(multi.M):
    fun=GridFunction(L)
    fun.vec[:]=0
    cutratio.append(fun)
    
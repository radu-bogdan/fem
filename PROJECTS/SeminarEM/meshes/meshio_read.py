import sys
sys.path.insert(0,'../../') # adds parent directory
sys.path.insert(0,'../CEM') # adds parent directory

import meshio
import numpy as np
import pde
import plotly.io as pio
pio.renderers.default = 'browser'


#####################################################################################################################
mesh = meshio.read("motor.vol",file_format = "netgen")
regions_2d = ('coil1', 'air', 'coil2', 'air', 'coil3', 'air', 'coil4', 'air', 'coil5', 'air', 'coil6', 'air', 'coil7', 'air', 'coil8', 'air', 'coil9', 'air', 'coil10', 'air', 'coil11', 'air', 'coil12', 'air', 'coil13', 'air', 'coil14', 'air', 'coil15', 'air', 'coil16', 'air', 'coil17', 'air', 'coil18', 'air', 'coil19', 'air', 'coil20', 'air', 'coil21', 'air', 'coil22', 'air', 'coil23', 'air', 'coil24', 'air', 'coil25', 'air', 'coil26', 'air', 'coil27', 'air', 'coil28', 'air', 'coil29', 'air', 'coil30', 'air', 'coil31', 'air', 'coil32', 'air', 'coil33', 'air', 'coil34', 'air', 'coil35', 'air', 'coil36', 'air', 'coil37', 'air', 'coil38', 'air', 'coil39', 'air', 'coil40', 'air', 'coil41', 'air', 'coil42', 'air', 'coil43', 'air', 'coil44', 'air', 'coil45', 'air', 'coil46', 'air', 'coil47', 'air', 'coil48', 'air', 'magnet1', 'magnet2', 'rotor_air', 'rotor_air', 'rotor_air', 'rotor_air', 'magnet3', 'magnet4', 'rotor_air', 'rotor_air', 'rotor_air', 'rotor_air', 'magnet5', 'magnet6', 'rotor_air', 'rotor_air', 'rotor_air', 'rotor_air', 'magnet7', 'magnet8', 'rotor_air', 'rotor_air', 'rotor_air', 'rotor_air', 'magnet9', 'magnet10', 'rotor_air', 'rotor_air', 'rotor_air', 'rotor_air', 'magnet11', 'magnet12', 'rotor_air', 'rotor_air', 'rotor_air', 'rotor_air', 'magnet13', 'magnet14', 'rotor_air', 'rotor_air', 'rotor_air', 'rotor_air', 'magnet15', 'magnet16', 'rotor_air', 'rotor_air', 'rotor_air', 'rotor_air', 'shaft_iron', 'rotor_iron', 'air_gap_stator', 'air_gap', 'air_gap_rotor', 'stator_iron')
regions_1d = ('default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'stator_inner', 'default', 'default', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'magnets_interface', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'rotor_inner', 'rotor_outer', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'stator_inner', 'default', 'default', 'stator_outer')

regions_2d_np = np.zeros((len(regions_2d),), dtype=object)
regions_2d_np[:] = list(regions_2d)

regions_1d_np = np.zeros((len(regions_1d),), dtype=object)
regions_1d_np[:] = list(regions_1d)

# air_indices = [i for i, x in enumerate(regions_2d) if x == "air"]
# air_indices = [i for i, x in enumerate(regions_2d) if x == "air"]
#####################################################################################################################



#####################################################################################################################
Mperp_mag1 = np.array([-0.507223091788922, 0.861814791678634])
Mperp_mag2 = np.array([-0.250741225095427, 0.968054150364350])
Mperp_mag3 = (-1)*np.array([-0.968055971101187, 0.250734195544481])
Mperp_mag4 = (-1)*np.array([-0.861818474866413, 0.507216833690415])
Mperp_mag5 = np.array([-0.861814791678634, -0.507223091788922])
Mperp_mag6 = np.array([-0.968054150364350, -0.250741225095427])
Mperp_mag7 = (-1)*np.array([-0.250734195544481, -0.968055971101187])
Mperp_mag8 = (-1)*np.array([-0.507216833690415, -0.861818474866413])
Mperp_mag9 = np.array([0.507223091788922, -0.861814791678634])
Mperp_mag10 = np.array([0.250741225095427, -0.968054150364350])
Mperp_mag11 = (-1)*np.array([0.968055971101187, -0.250734195544481])
Mperp_mag12 = (-1)*np.array([0.861818474866413, -0.507216833690415])
Mperp_mag13 = np.array([0.861814791678634, 0.507223091788922])
Mperp_mag14 = np.array([0.968054150364350, 0.250741225095427])
Mperp_mag15 = (-1)*np.array([0.250734195544481, 0.968055971101187])
Mperp_mag16 = (-1)*np.array([0.507216833690415, 0.861818474866413])

Mperp_mag = np.c_[Mperp_mag1,Mperp_mag2, Mperp_mag3, Mperp_mag4, Mperp_mag5, Mperp_mag6, Mperp_mag7, Mperp_mag8,
                  Mperp_mag9,Mperp_mag10,Mperp_mag11,Mperp_mag12,Mperp_mag13,Mperp_mag14,Mperp_mag15,Mperp_mag16]
m = np.c_[Mperp_mag[1,:],-Mperp_mag[0,:]].T
#####################################################################################################################




#####################################################################################################################
def f48(s):
    return (s-1)%48

offset = 0
polepairs  = 4
gamma_correction_model = -30.0
gamma = 40.0
gamma_correction_timestep = -1
phi0 = (gamma + gamma_correction_model + gamma_correction_timestep * polepairs) * np.pi/180.0


area_coils_UPlus  = np.r_[f48(offset+1) , f48(offset+2)]
area_coils_VMinus = np.r_[f48(offset+3) , f48(offset+4)]
area_coils_WPlus  = np.r_[f48(offset+5) , f48(offset+6)]
area_coils_UMinus = np.r_[f48(offset+7) , f48(offset+8)]
area_coils_VPlus  = np.r_[f48(offset+9) , f48(offset+10)]
area_coils_WMinus = np.r_[f48(offset+11), f48(offset+12)]

for k in range(1,4):
    area_coils_UPlus = np.r_[area_coils_UPlus, f48(k*12+offset+1)]
    area_coils_UPlus = np.r_[area_coils_UPlus, f48(k*12+offset+2)]
    
    area_coils_VMinus = np.r_[area_coils_VMinus, f48(k*12+offset+3)]
    area_coils_VMinus = np.r_[area_coils_VMinus, f48(k*12+offset+4)]
    
    area_coils_WPlus = np.r_[area_coils_WPlus, f48(k*12+offset+5)]
    area_coils_WPlus = np.r_[area_coils_WPlus, f48(k*12+offset+6)]
    
    area_coils_UMinus = np.r_[area_coils_UMinus, f48(k*12+offset+7)]
    area_coils_UMinus = np.r_[area_coils_UMinus, f48(k*12+offset+8)]
    
    area_coils_VPlus = np.r_[area_coils_VPlus, f48(k*12+offset+9)]
    area_coils_VPlus = np.r_[area_coils_VPlus, f48(k*12+offset+10)]
    
    area_coils_WMinus = np.r_[area_coils_WMinus, f48(k*12+offset+11)]
    area_coils_WMinus = np.r_[area_coils_WMinus, f48(k*12+offset+12)]
    
    
I0peak = 1555.63491861 ### *1.5
phase_shift_I1 = 0.0
phase_shift_I2 = 2/3*np.pi#4/3*pi
phase_shift_I3 = 4/3*np.pi#2/3*pi

I1c = I0peak * np.sin(phi0 + phase_shift_I1)
I2c = (-1)* I0peak * np.sin(phi0 + phase_shift_I2)
I3c = I0peak * np.sin(phi0 + phase_shift_I3)

areaOfOneCoil = 0.00018053718538758062

UPlus  =  I1c* 2.75 / areaOfOneCoil
VMinus = -I2c* 2.75 / areaOfOneCoil

WPlus  =  I3c* 2.75 / areaOfOneCoil
UMinus = -I1c* 2.75 / areaOfOneCoil

VPlus  =  I2c* 2.75 / areaOfOneCoil
WMinus = -I3c* 2.75 / areaOfOneCoil
    
j3 = np.zeros(48)
j3[area_coils_UPlus]  = UPlus
j3[area_coils_VMinus] = VMinus
j3[area_coils_WPlus]  = WPlus
j3[area_coils_UMinus] = UMinus
j3[area_coils_VPlus]  = VPlus
j3[area_coils_WMinus] = WMinus



#####################################################################################################################
p = mesh.points
t = np.c_[mesh.cells_dict['triangle'],
          mesh.cell_data_dict['netgen:index']['triangle']-1]
e = np.c_[mesh.cells_dict['line'],
          mesh.cell_data_dict['netgen:index']['line']-1]
q = np.empty(0)

import scipy.io
scipy.io.savemat('motor.mat', {"t" : t.T+1,
                               "e" : e.T+1,
                               "p" : p.T,
                               "regions_2d" : regions_2d_np,
                               "regions_1d" : regions_1d_np,
                               "m" : m,
                               "j3" : j3},do_compression=True)
#####################################################################################################################



#####################################################################################################################
np.savez_compressed('motor.npz', p=p.T, e=e.T, t=t.T, regions_2d=regions_2d_np, regions_1d=regions_1d_np, m=m, j3=j3)
#####################################################################################################################




MESH = pde.mesh(p,e,t,q)

ind_air_all = np.flatnonzero(np.core.defchararray.find(regions_2d,'air')!=-1)
ind_stator_rotor = np.flatnonzero(np.core.defchararray.find(regions_2d,'iron')!=-1)
ind_magnet = np.flatnonzero(np.core.defchararray.find(regions_2d,'magnet')!=-1)
ind_coil = np.flatnonzero(np.core.defchararray.find(regions_2d,'coil')!=-1)
ind_shaft = np.flatnonzero(np.core.defchararray.find(regions_2d,'shaft')!=-1)

trig_air_all = np.where(np.isin(t[:,3],ind_air_all))
trig_stator_rotor = np.where(np.isin(t[:,3],ind_stator_rotor))
trig_magnet = np.where(np.isin(t[:,3],ind_magnet))
trig_coil = np.where(np.isin(t[:,3],ind_coil))
trig_shaft = np.where(np.isin(t[:,3],ind_shaft))

vek = np.zeros(MESH.nt)
vek[trig_air_all] = 1
vek[trig_magnet] = 2
vek[trig_coil] = 3
vek[trig_stator_rotor] = 4
vek[trig_shaft] = 3.6

# fig = MESH.pdemesh()
fig = MESH.pdesurf_hybrid(dict(trig = 'P0',quad = 'Q0',controls = 0), vek, u_height=0)
fig.show()
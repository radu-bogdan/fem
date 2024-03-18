print('t13_geo')

import ngsolve as ng
import netgen.occ as occ
import time
import matplotlib.pyplot as plt
import numpy as np

################# Bottom part #################

p1 = (0,0,0)
p2 = (1.5,0,0)
p3 = (1.5,0.25,0)
p4 = (0,0.25,0)

e1 = occ.Segment(p1,p2)
e2 = occ.Segment(p2,p3)
e3 = occ.Segment(p3,p4)
e4 = occ.Segment(p4,p1)

bottom = occ.Face(occ.Wire([e1,e2,e3,e4]))


################# Top part #################

p5 = (0,0.25+0.025,0)
p6 = (1.5,0.25+0.025,0)
p7 = (1.5,0.25+0.025+0.75,0)
p8 = (0,0.25+0.025+0.75,0)

e5 = occ.Segment(p5,p6)
e6 = occ.Segment(p6,p7)
e7 = occ.Segment(p7,p8)
e8 = occ.Segment(p8,p5)

top = occ.Face(occ.Wire([e5,e6,e7,e8]))

e1 = occ.Segment(p5,p6)
e2 = occ.Segment(p6,p7)
e3 = occ.Segment(p7,p8)
e4 = occ.Segment(p8,p5)



################# New #################

points = [( 0.25, 0,    0), ( 0.25, 0.50, 0), ( 0.50, 0.50, 0), ( 0.75, 0,    0),
          ( 0.75, 0.75, 0), ( 0.50, 0,    0), (-0.75, 0.75, 0), (-0.75, 0    ,0),
          (-0.25, 0    ,0), (-0.25, 0.5  ,0), (-0.50, 0.5  ,0), (-0.50, 0    ,0),
          ( 0.75,-0.025,0), (-0.75,-0.025,0), (-0.75,-0.275,0), ( 0.75,-0.275,0),
          (-1.25, 1.25 ,0), ( 1.25, 1.25 ,0), ( 1.25,-0.75 ,0), (-1.25,-0.75 ,0)]

# edges = np.array([[4,3],[0,1],[2,1],[2,5],[5,3],[6,7],[8,9],[10,9],[10,11],[11,7],[13,12],[14,15],[13,14],[12,15],[16,17],[17,18],[18,19],[19,16],[11,8],[0,5],[6,4],[8,0]])
edges = np.array([[9,10],[10,11],[11,8],[8,9], [2,1],[1,0],[0,5],[5,2], [17,16],[16,19],[19,18],[18,17], [3,7],[7,6],[6,4],[4,3], [15,14],[14,13],[13,12],[12,15]])
faces = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]])

e = {}
for i,j in enumerate(edges): e[i] = occ.Segment(points[j[0]],points[j[1]])

f = {}
for i,j in enumerate(faces): f[i] = occ.Face(occ.Wire([e[j[0]],e[j[1]],e[j[2]],e[j[3]]]))





fig, ax = plt.subplots()
npp = np.array(points)
ax.scatter(npp[:,0],npp[:,1])

n = np.arange(npp.shape[0])
for i, txt in enumerate(n):
    ax.annotate(txt, (npp[i,0], npp[i,1]))
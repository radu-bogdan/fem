import netgen.occ as occ

points = [( 0.25, 0,    0), ( 0.25, 0.50, 0), ( 0.50, 0.50, 0), ( 0.75, 0,    0),
          ( 0.75, 0.75, 0), ( 0.50, 0,    0), (-0.75, 0.75, 0), (-0.75, 0    ,0),
          (-0.25, 0    ,0), (-0.25, 0.5  ,0), (-0.50, 0.5  ,0), (-0.50, 0    ,0),
          ( 0.75,-0.025,0), (-0.75,-0.025,0), (-0.75,-0.275,0), ( 0.75,-0.275,0),
          (-1.25, 1.25 ,0), ( 1.25, 1.25 ,0), ( 1.25,-0.75 ,0), (-1.25,-0.75 ,0)]

edges = [[9,10],[10,11],[11,8],[8,9], [2,1],[1,0],[0,5],[5,2], [17,16],[16,19],[19,18],[18,17], [3,7],[7,6],[6,4],[4,3], [15,14],[14,13],[13,12],[12,15], [12,3],[7,13]]
faces = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[18,20,13,21]]

e = {}
for i,j in enumerate(edges): e[i] = occ.Segment(points[j[0]],points[j[1]])

f = {}
for i,j in enumerate(faces): f[i] = occ.Face(occ.Wire([e[j[0]],e[j[1]],e[j[2]],e[j[3]]]))

geo = occ.Glue(list(f.values()))

geo.faces[0].name = 'coil_plus'; geo.faces[0].col = (1,0.5,0)
geo.faces[1].name = 'coil_minus'; geo.faces[1].col = (1,0.5,0)
geo.faces[2].name = 'air';   geo.faces[2].col = (0,0.5,1)
geo.faces[3].name = 'iron';  geo.faces[3].col = (0,0.5,0)
geo.faces[4].name = 'air';   geo.faces[4].col = (0,0.5,1)
geo.faces[5].name = 'iron';  geo.faces[5].col = (0,0.5,0)

geo.edges[8].name = 'outer'
geo.edges[9].name = 'outer'
geo.edges[10].name = 'outer'
geo.edges[11].name = 'outer'

geoOCC = occ.OCCGeometry(geo, dim = 2)
geoOCCmesh = geoOCC.GenerateMesh(maxh = 0.1)
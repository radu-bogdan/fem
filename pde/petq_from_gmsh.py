# Code	Type	Description
# 1	2-node line	
# 2	3-node triangle	
# 3	4-node quadrangle	
# 4	4-node tetrahedron	
# 5	8-node hexahedron	
# 6	6-node prism	
# 7	5-node pyramid	
# 8	3-node second order line	(2 nodes associated with the vertices and 1 with the edge).
# 9	6-node second order triangle	(3 nodes associated with the vertices and 3 with the edges).
# 10	9-node second order quadrangle	(4 nodes associated with the vertices, 4 with the edges and 1 with the face).
# 11	10-node second order tetrahedron	(4 nodes associated with the vertices and 6 with the edges).
# 12	27-node second order hexahedron	(8 nodes associated with the vertices, 12 with the edges, 6 with the faces and 1 with the volume).
# 13	18-node second order prism	(6 nodes associated with the vertices, 9 with the edges and 3 with the quadrangular faces).
# 14	14-node second order pyramid	(5 nodes associated with the vertices, 8 with the edges and 1 with the quadrangular face).
# 15	1-node point	
# 16	8-node second order quadrangle	(4 nodes associated with the vertices and 4 with the edges).
# 17	20-node second order hexahedron	(8 nodes associated with the vertices and 12 with the edges).
# 18	15-node second order prism	(6 nodes associated with the vertices and 9 with the edges).
# 19	13-node second order pyramid	(5 nodes associated with the vertices and 8 with the edges).
# 20	9-node third order incomplete triangle	(3 nodes associated with the vertices, 6 with the edges)
# 21	10-node third order triangle	(3 nodes associated with the vertices, 6 with the edges, 1 with the face)
# 22	12-node fourth order incomplete triangle	(3 nodes associated with the vertices, 9 with the edges)
# 23	15-node fourth order triangle	(3 nodes associated with the vertices, 9 with the edges, 3 with the face)
# 24	15-node fifth order incomplete triangle	(3 nodes associated with the vertices, 12 with the edges)
# 25	21-node fifth order complete triangle	(3 nodes associated with the vertices, 12 with the edges, 6 with the face)
# 26	4-node third order edge	(2 nodes associated with the vertices, 2 internal to the edge)
# 27	5-node fourth order edge	(2 nodes associated with the vertices, 3 internal to the edge)
# 28	6-node fifth order edge	(2 nodes associated with the vertices, 4 internal to the edge)
# 29	20-node third order tetrahedron	(4 nodes associated with the vertices, 12 with the edges, 4 with the faces)
# 30	35-node fourth order tetrahedron	(4 nodes associated with the vertices, 18 with the edges, 12 with the faces, 1 in the volume)
# 31	56-node fifth order tetrahedron	(4 nodes associated with the vertices, 24 with the edges, 24 with the faces, 4 in the volume)
# 92	64-node third order hexahedron	(8 nodes associated with the vertices, 24 with the edges, 24 with the faces, 8 in the volume)
# 93	125-node fourth order hexahedron	(8 nodes associated with the vertices, 36 with the edges, 54 with the faces, 27 in the volume)

import gmsh
import numpy
import numpy.matlib
# numpy.set_printoptions(edgeitems=30, linewidth = 1000000)

def petq_from_gmsh(filename,hmax=1):
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0) # Supress the output of GMSH
    gmsh.option.setNumber("Mesh.MeshSizeFactor", hmax)
    gmsh.option.setNumber("Mesh.MeshSizeMin", hmax)
    gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
    gmsh.option.setNumber("Mesh.Algorithm", 2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    
    gmsh.open(filename)
    gmsh.model.mesh.generate(2)
    
    entities = gmsh.model.getEntities()

    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim=-1,tag=-1)
    p = nodeCoords.reshape(int(nodeCoords.shape[0]/3),3,order='C')
    p = p[:,[0,1]]

    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=-1,tag=-1)
    physical_groups = gmsh.model.getPhysicalGroups(dim=1)


    e  = numpy.empty(shape=(1,0),dtype=numpy.uint64)
    region  = numpy.empty(shape=(1,0),dtype=numpy.uint64)
    for i in range(len(physical_groups)):
        _, _, elemNodeTags_1d = gmsh.model.mesh.getElements(dim=1,tag=gmsh.model.getEntitiesForPhysicalGroup(dim=1,tag=physical_groups[i][1])[0])
        e  = numpy.concatenate([e,elemNodeTags_1d[0][None]],axis=1)
        new_region = numpy.matlib.repmat(physical_groups[i][1],1,int(elemNodeTags_1d[0].size/2)).astype(numpy.uint64)
        region = numpy.concatenate([region,new_region],axis=1)
    e = (e-1).reshape(int(e[0].shape[0]/2),2,order='C')
    e = numpy.concatenate([e,region.T],axis=1)

    # initialize as empty incase there are not triangles/quads.
    t = numpy.array([],dtype=numpy.uint64)
    q = numpy.array([],dtype=numpy.uint64)

    for i in range(elemTypes.size):
        # if elemTypes[i]==1: # 2-node line (see comment at top)
        #     e_all = (elemNodeTags[i]-1).reshape(int(elemNodeTags[i].shape[0]/2),2,order='C')
        if elemTypes[i]==2: # 3-node triangle (see comment at top)
            t = (elemNodeTags[i]-1).reshape(int(elemNodeTags[i].shape[0]/3),3,order='C')
        if elemTypes[i]==3: # 4-node quadrangle (see comment at top)
            q = (elemNodeTags[i]-1).reshape(int(elemNodeTags[i].shape[0]/4),4,order='C')

    gmsh.clear()
    gmsh.finalize()
    print('Generated mesh with ' + str(p.shape[0]) + ' points, ' 
                                 + str(e.shape[0]) + ' boundary edges, ' 
                                 + str(t.shape[0]) + ' triangles, ' 
                                 + str(q.shape[0]) + ' quadrilaterals.')
    return p,e,t,q

# if __name__ == "__main__":
#     p,e,t,q = petq_from_gmsh('mesh_new.geo',hmax=0.8)
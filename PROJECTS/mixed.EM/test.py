import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

x = np.array( [
[ [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]  ],  #element 0
[ [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]  ],  #element 1
[ [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]  ],  #element 2
] )

y = np.array( [
[ [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]  ],  #element 0
[ [0.0, 1.0], [0.5, 1.5], [1.0, 2.0]  ],  #element 1
[ [1.0, 1.0], [1.5, 1.5], [2.0, 2.0]  ],  #element 2
] )

z = np.array( [
[ [0.0, 0.5], [0.0, 0.8], [0.0, 1.0]  ],  #element 0
[ [0.3, 1.0], [0.6, 1.2], [0.8, 1.3]  ],  #element 1
[ [1.2, 1.5], [1.3, 1.4], [1.5, 1.7]  ],  #element 2
] )



global_num_pts =  z.size
global_x = np.zeros( global_num_pts )
global_y = np.zeros( global_num_pts )
global_z = np.zeros( global_num_pts )
global_triang_list = list()

offset = 0;
num_triangles = 0;

#process triangulation element-by-element
for k in range(z.shape[0]):
    points_x = x[k,...].flatten()
    points_y = y[k,...].flatten()
    z_element = z[k,...].flatten()
    num_points_this_element = points_x.size

    #auto-generate Delauny triangulation for the element, which should be flawless due to quadrilateral element shape
    triang = tri.Triangulation(points_x, points_y)
    global_triang_list.append( triang.triangles + offset ) #offseting triangle indices by start index of this element

    #store results for this element in global triangulation arrays
    global_x[offset:(offset+num_points_this_element)] = points_x
    global_y[offset:(offset+num_points_this_element)] = points_y
    global_z[offset:(offset+num_points_this_element)] = z_element

    num_triangles += triang.triangles.shape[0]
    offset += num_points_this_element


#go back and turn all of the triangle indices into one global triangle array
offset = 0
global_triang = np.zeros( (num_triangles, 3) )
for t in global_triang_list:
    global_triang[ offset:(offset+t.shape[0] )] = t
    offset += t.shape[0]

plt.figure()
plt.gca().set_aspect('equal')

plt.tripcolor(global_x, global_y, global_triang, global_z, shading='gouraud' )
# plt.tricontour(global_x, global_y, global_triang, global_z )
plt.triplot(global_x, global_y, global_triang, 'go-') #plot just the triangle mesh

plt.xlim((-0.25, 3.25))
plt.ylim((-0.25, 2.25))
plt.show()
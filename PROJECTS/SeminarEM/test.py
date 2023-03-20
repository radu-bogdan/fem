from math import pi, cos
import pygmsh

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_max = 0.1
    r = 0.5
    disks = [
        geom.add_disk([-0.5 * cos(7 / 6 * pi), -0.25], 1.0),
        geom.add_disk([+0.5 * cos(7 / 6 * pi), -0.25], 1.0),
        geom.add_disk([0.0, 0.5], 1.0),
    ]
    geom.boolean_intersection(disks)

    mesh = geom.generate_mesh()
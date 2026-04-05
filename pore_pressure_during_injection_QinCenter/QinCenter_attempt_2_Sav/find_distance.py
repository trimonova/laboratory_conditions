import shapely.geometry as geom
import numpy as np
import matplotlib.pyplot as plt

coords = [(3, 1), (3, 3), (6, 3), (6, 1), (3, 1)]

line = geom.LineString(coords)
point = geom.Point((3, 2))

plt.plot(line)
#plt.plot(point)
plt.show()
# Note that "line.distance(point)" would be identical
print(point.distance(line))

from shapely.geometry import Point, Polygon
print(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]).contains(Point(0.5, 0.001)))

coord_matrix_rad = [[0]*5]*3
coord_matrix_rad[1][1] = 5
print(coord_matrix_rad)
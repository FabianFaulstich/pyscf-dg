import sys
sys.path.append('../../src')
import dg_tools as dg

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,ConvexHull

atoms = np.array([[-3, -.1], [-1, .1],
                      [1, -.2], [3, .2]])
x_max =  6
x_min = -6
y_max =  6
y_min = -6



atoms = dg.tile_atoms(atoms,x_max - x_min, y_max - y_min)
voronoi  = Voronoi(atoms)

edges = np.array(voronoi.ridge_vertices)
center = atoms.mean(axis=0)

V_net = []
for k, v in enumerate(voronoi.vertices):
    if x_min <= v[0] <= x_max and y_min <= v[1] <= y_max:
        vert =[]
        vert.append(k)
        vert.append(v.tolist())
        c = []
        # Checking for connections within simulation box
        for j, v_n in enumerate(V_net):
            if any(np.equal([v_n[0],k],edges).all(1)):
                c.append(j)
                v_n[2].append(len(V_net))
            elif any(np.equal([k,v_n[0]],edges).all(1)):
                c.append(j)
                v_n[2].append(len(V_net))
        vert.append(c)
        V_net.append(vert)

VV = [[vert[1],vert[2]] for vert in V_net]
for j, v in enumerate(V_net):
    if np.count_nonzero(edges == v[0]) != len(v[2]): 
        # Add boundary points:
        # computing connections out of the simulation box (c)
        c = edges[edges[:,0] == v[0]]
        c = np.vstack((c,edges[edges[:,1] == v[0]]))
        for k in v[2]:
            c = c[c[:,0] != V_net[k][0]]
            c = c[c[:,1] != V_net[k][0]]
        for outer in c:
            # computing boundary point
            e_idx = np.where((edges[:,0] == outer[0]) & (edges[:,1]== outer[1]))[0][0]
            v_out = voronoi.vertices[outer[outer != v[0]]][0]
            a_idx = voronoi.ridge_points[e_idx]
            bp = dg.get_boundary_point(v[1], atoms[a_idx[0]],atoms[a_idx[1]], center, x_min, x_max, y_min, y_max, v_out)
            VV.append([bp.tolist(),[j]])
            VV[j][1].append(len(VV))
                        
VERTEX = np.array([vert[0] for vert in VV])
print(VERTEX)
#exit()
# plotting conections between two voronoi vertices: 
for rv in voronoi.ridge_vertices:
    rv = np.asarray(rv)
    if np.all(rv >= 0):
        plt.plot(voronoi.vertices[rv,0], voronoi.vertices[rv,1], 'k-')

# plotting connextions from box limit with voronoi 
center = atoms.mean(axis=0)
plt.plot(center[0],center[1],'r+')
for pointidx, rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
    rv = np.asarray(rv)
    if np.any(rv < 0):
        i = rv[rv >= 0][0]
        t = atoms[pointidx[1]]- atoms[pointidx[0]]
        t = t/np.linalg.norm(t)
        n = np.array([-t[1],t[0]])
        midpoint = atoms[pointidx].mean(axis=0)
        far_point = voronoi.vertices[i] + np.sign(np.dot(midpoint-center, n)) * n * 100
        plt.plot([voronoi.vertices[i,0],far_point[0]], [voronoi.vertices[i,1], far_point[1]], 'k-')

# plotting voronoi vertices: 
plt.plot(voronoi.vertices[:,0], voronoi.vertices[:,1],'D')  # poltting voronoi vertices
plt.plot(atoms[:,0], atoms[:,1], 'bo')
plt.plot(VERTEX[:,0], VERTEX[:,1], 'kD')
plt.plot([[x_min,y_min],[x_min,y_max],[y_max,x_max],[x_max,y_min]],[[x_min,y_max],[y_max,x_max],[x_max,y_min],[x_min,y_min]],'r-')
plt.xlim(-18,18)
plt.ylim(-18,18)
plt.show()


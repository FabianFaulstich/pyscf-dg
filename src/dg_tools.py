import math
import numpy as np
from scipy.spatial import Delaunay, Voronoi

def unfold(m,n):
    l = m % n
    k = (m-l)/n
    return int(l), int(k)

def unfold_sym(n):
    k = 1 
    while n>(1+k)*k/2.0:
        k = k+1
    l = n - (k-1)*k/2.0
    return int(k-1), int(l-1)

def get_red_idx(n):
    idx_sym = np.zeros(int(n*(n+1)/2));
    count   = 0;
    for j in range(n):
        l = n-j;
        idx_sym[count:count+l] = np.arange(n*j+j, n*j+n);
        count = count + l;
    return idx_sym.astype(int)

# for voronoi
def in_hull(p, hull):
    """
    Test if points in p are in hull
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def get_V_cells(V_net , atoms):
    """ 
    Only 2D
    V_net:
    atoms:
    """
    voronoi_cells = []
    for i, atom in enumerate(atoms):
        cell = []
        cell.append(get_cell(atom, V_net))

        exit()

        break

def get_cell(atom, V_net):
    
    verts = np.array([elem[0] for elem in V_net])
    idx   = np.argmin((np.sum(np.abs(verts-atom)**2,axis=-1)**(1./2)))
    vert  = verts[idx]
    con   = V_net[idx][1] 
    
    get_angle(atom, vert, V_net[con[0]][0])

def get_angle(atom, vert, vert_1):
    """ 
    Returns clockwise measured angle
    """
    #print("atom: ", atom)
    #print("Vertex: ", vert)
    #print("Connected Vertex: ", vert_1)

    v_0 = atom - vert
    v_1 = vert_1 - vert

    v_dot = np.dot(v_0,v_1)
    v_det = v_0[0]*v_1[1]-v_0[1]*v_1[0]
    # clockwise angle
    angle = math.atan2(v_det,v_dot)*180/np.pi
    if angle < 0:
        angle = 360 + angle
    return angle
    #print(v_0)
    #print(v_1)
    #print(angle)


def get_V_net(atoms, x_min, x_max, y_min, y_max):
    """
    Only 2D for now
    atoms:
    box parameters:
    """

    # Get "innter" vertices from scipy
    vor  = Voronoi(atoms)
    edge = vor.ridge_vertices
    vert = vor.vertices

    # Compute vertices on the boundary
    center = atoms.mean(axis=0)
    for j, (pointidx, rv) in enumerate(zip(vor.ridge_points, edge)):
        rv = np.asarray(rv)
        if np.any(rv < 0):
            i = rv[rv >= 0][0]
            t = atoms[pointidx[1]]- atoms[pointidx[0]]
            t = t/np.linalg.norm(t)
            n = np.array([-t[1],t[0]])
            midpoint = atoms[pointidx].mean(axis=0)
            normal = np.sign(np.dot(midpoint-center, n))*n
            print(normal)
            if normal[0] < 0 and normal[1] < 0:
                dx = vert[i][0]-x_min
                dy = vert[i][1]-y_min
                scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
            elif normal[0] > 0 and normal[1] > 0:
                dx = x_max-vert[i][0]
                dy = y_max-vert[i][1]
                scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
            elif normal[0] < 0 and normal[1] > 0:
                dx = vert[i][0]-x_min
                dy = y_max-vert[i][1]
                scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
            elif normal[0] > 0 and normal[1] < 0:
                dx = x_max-vert[i][0]
                dy = vert[i][1]-y_min
                scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
            elif normal[0] == 0 and normal[1] < 0:
                dy = vert[i][1]-y_min
                scalar = np.sign(normal[1])*dy/normal[1] 
            elif normal[0] == 0 and normal[1] > 0:
                dy = y_max-vert[i][1]
                scalar = np.sign(normal[1])*dy/normal[1] 
            elif normal[0] < 0 and normal[1] == 0:
                dx = vert[i][0]-x_min
                scalar = np.sign(normal[0])*dx/normal[0] 
            elif normal[0] > 0 and normal[1] == 0:
                dx = x_max-vert[i][0]
                scalar = np.sign(normal[0])*dx/normal[0]

            boundary_point = vert[i] + np.sign(np.dot(midpoint-center, n))*n*scalar
            vert = np.vstack((vert, boundary_point))
            edge[j][np.argmin(rv)] = len(vert) #only 2D for now

    # Add boundary vertices and edges
    vert = np.vstack((vert, np.array([x_min, y_min])))
    vert = np.vstack((vert, np.array([x_max, y_min])))
    vert_s = np.array([v for v in vert if v[1] == y_min])
    vert_s = np.array(sorted(vert_s, key=lambda x: x[0]))
    for i, v in enumerate(vert_s[1:]):
        idx_0 = np.where((vert[:,0] == vert_s[i][0]) & (vert[:,1]== vert_s[i][1]))[0][0]
        idx_1 = np.where((vert[:,0] == v[0]) & (vert[:,1]==v[1]))[0][0]
        edge.append([idx_0, idx_1])

    vert = np.vstack((vert, np.array([x_max, y_max])))
    vert_s = np.array([v for v in vert if v[0] == x_max])
    vert_s = np.array(sorted(vert_s, key=lambda x: x[1]))
    for i, v in enumerate(vert_s[1:]):
        idx_0 = np.where((vert[:,0] == vert_s[i][0]) & (vert[:,1]== vert_s[i][1]))[0][0]
        idx_1 = np.where((vert[:,0] == v[0]) & (vert[:,1]==v[1]))[0][0]
        edge.append([idx_0, idx_1])

    vert = np.vstack((vert, np.array([x_min, y_max])))
    vert_s = np.array([v for v in vert if v[1] == y_max])
    vert_s = np.array(sorted(vert_s, key=lambda x: x[0]))
    for i, v in enumerate(vert_s[1:]):
        idx_0 = np.where((vert[:,0] == vert_s[i][0]) & (vert[:,1]== vert_s[i][1]))[0][0]
        idx_1 = np.where((vert[:,0] == v[0]) & (vert[:,1]==v[1]))[0][0]
        edge.append([idx_0, idx_1])

    vert_s = np.array([v for v in vert if v[0] == x_min])
    vert_s = np.array(sorted(vert_s, key=lambda x: x[1]))
    for i, v in enumerate(vert_s[1:]):
        idx_0 = np.where((vert[:,0] == vert_s[i][0]) & (vert[:,1]== vert_s[i][1]))[0][0]
        idx_1 = np.where((vert[:,0] == v[0]) & (vert[:,1]==v[1]))[0][0]
        edge.append([idx_0, idx_1])

    # create Voronoi network

    edge_inv = [[e[1],e[0]] for e in edge]
    edge = edge + [[e[1],e[0]] for e in edge]
    edge = np.array(sorted(edge, key=lambda x: x[0]))
    V_net = []
    for j, v in enumerate(vert):
        idx = np.where((edge[:,0] == j))[0]
        V_net.append([v, edge[idx][:,1]])

    return V_net

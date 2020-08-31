import math
import numpy as np
from scipy.spatial import Delaunay, Voronoi
import matplotlib.pyplot as plt
from numpy import linalg as la

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
        voronoi_cells.append(get_cell(atom, V_net))
    return np.array(voronoi_cells)   

def get_cell(atom, V_net):
    
    verts = np.array([elem[0] for elem in V_net])
    idx   = np.argmin((np.sum(np.abs(verts-atom)**2,axis=-1)**(1./2)))
    vert  = verts[idx]
    con   = V_net[idx][1]
    cell  = [vert]

    idx_new, vert_new = get_next_vert(atom, idx, V_net)
    cell.append(vert_new)
    es = 0

    # small systems, no more than 100 vertices!
    while (cell[-1][0] != vert[0] or cell[-1][1] != vert[1]) and es < 100:
        idx_new, vert_new = get_next_vert(cell[-2], idx_new, V_net)
        cell.append(vert_new)
        es += 1

    return np.array(cell)

def get_next_vert(atom, idx, V_net):
    vert = V_net[idx][0]
    con  = V_net[idx][1]
    angle = []
    vertex = []
    cc = []
    for c in con:
        if not(atom[0] == V_net[c][0][0] and atom[1] == V_net[c][0][1]):
            angle.append(get_angle(atom, vert, V_net[c][0]))
            vertex.append(V_net[c][0])
            cc.append(c) 
    angle = np.array(angle)
    idx_min = np.argmin(angle)
    return cc[idx_min], vertex[idx_min]

def get_angle(atom, vert, vert_1):
    """ 
    Returns angle between atom-vert and vert-vert_1
    """
    v_0 = atom - vert
    v_1 = vert_1 - vert
    
    v_dot = np.dot(v_0,v_1)
    v_det = v_0[0]*v_1[1]-v_0[1]*v_1[0]
    #angle
    angle = math.atan2(v_det,v_dot)*180/np.pi
    if angle < 0:
        angle = 360 + angle
    return angle

def tile_atoms(atoms, Dx, Dy):
    """Only 2D for now
    """
    N = len(atoms)
    atoms_per = np.zeros((9*N,2))
    # upper left 
    atoms_per[:N, 0] = atoms[:,0] - Dx 
    atoms_per[:N, 1] = atoms[:,1] + Dy

    # upper mid
    atoms_per[N:2*N, 0] = atoms[:,0]
    atoms_per[N:2*N, 1] = atoms[:,1] + Dy

    # upper right
    atoms_per[2*N:3*N, 0] = atoms[:,0] + Dx
    atoms_per[2*N:3*N, 1] = atoms[:,1] + Dy

    # left 
    atoms_per[3*N:4*N, 0] = atoms[:,0] - Dx
    atoms_per[3*N:4*N, 1] = atoms[:,1]

    # mid
    atoms_per[4*N:5*N, 0] = atoms[:,0]
    atoms_per[4*N:5*N, 1] = atoms[:,1]

    # right
    atoms_per[5*N:6*N, 0] = atoms[:,0] + Dx
    atoms_per[5*N:6*N, 1] = atoms[:,1]

    # lower left 
    atoms_per[6*N:7*N,0] = atoms[:,0] - Dx
    atoms_per[6*N:7*N,1] = atoms[:,1] - Dy

    # lower mid
    atoms_per[7*N:8*N,0] = atoms[:,0] 
    atoms_per[7*N:8*N,1] = atoms[:,1] - Dy

    # lower right
    atoms_per[8*N:9*N, 0] = atoms[:,0] + Dx
    atoms_per[8*N:9*N, 1] = atoms[:,1] - Dy
    
    return atoms_per

def get_V_net_per(atoms, x_min, x_max, y_min, y_max):
    """Only 2D for now
    atoms:
    box parameters:
    """

    # tiling atoms:
    atoms = tile_atoms(atoms,x_max - x_min, y_max - y_min)
    vor   = Voronoi(atoms)
    edge  = np.array(vor.ridge_vertices)
    center = atoms.mean(axis=0)
    V_net = []

    # fetch inner points
    # Rounding to ensure to captur all vertices (C2H4)
    vor.vertices = np.around(vor.vertices, decimals=8)
    for k, v in enumerate(vor.vertices):
        if x_min <= v[0] <= x_max and y_min <= v[1] <= y_max:
            vert =[] 
            vert.append(k)
            vert.append(v.tolist())
            c = []
            for j, v_n in enumerate(V_net):
                if any(np.equal([v_n[0],k],edge).all(1)):
                    c.append(j)
                    v_n[2].append(len(V_net))
                elif any(np.equal([k,v_n[0]],edge).all(1)):
                    c.append(j)
                    v_n[2].append(len(V_net))
            vert.append(c)
            V_net.append(vert)
    
    V_net_per = [[vert[1],vert[2]] for vert in V_net]
    
    # Adding points on boundary
    for j, v in enumerate(V_net):
        if np.count_nonzero(edge == v[0]) != len(v[2]):
            # Add boundary points:
            # computing connections out of the simulation box (c)
            c = edge[edge[:,0] == v[0]]
            c = np.vstack((c,edge[edge[:,1] == v[0]]))
            for k in v[2]:
                c = c[c[:,0] != V_net[k][0]]
                c = c[c[:,1] != V_net[k][0]]
            for outer in c:
                # computing boundary point
                e_idx = np.where((edge[:,0] == outer[0]) & (edge[:,1]== outer[1]))[0][0]
                v_out = vor.vertices[outer[outer != v[0]]][0]
                a_idx = vor.ridge_points[e_idx]
                bp = get_boundary_point(v[1], atoms[a_idx[0]],atoms[a_idx[1]], center, x_min, x_max, y_min, y_max, v_out)
                if not any(np.equal(bp,[v[0] for v in V_net_per]).all(1)):
                    V_net_per.append([bp.tolist(),[j]])
                    V_net_per[j][1].append(len(V_net_per)-1)
    
    # Add boundary vertices and edges

    if not any(np.equal([x_min,y_min],[v[0] for v in V_net_per]).all(1)):
        V_net_per.append([[x_min, y_min],[]])
    if not any(np.equal([x_max,y_min],[v[0] for v in V_net_per]).all(1)):
        V_net_per.append([[x_max, y_min],[]])

    vert   = np.array([v[0] for v in V_net_per])
    vert_s = np.array([v for v in vert if np.abs(v[1] - y_min)<1e-8])
    vert_s = np.array(sorted(vert_s, key=lambda x: x[0]))
   
    for i, v in enumerate(vert_s[1:]):
        idx_0 = np.where((vert[:,0] == vert_s[i][0]) & (vert[:,1]== vert_s[i][1]))[0][0]
        idx_1 = np.where((vert[:,0] == v[0]) & (vert[:,1]==v[1]))[0][0]
        V_net_per[idx_0][1].append(idx_1)
        V_net_per[idx_1][1].append(idx_0)
    
    if not any(np.equal([x_max,y_max],[v[0] for v in V_net_per]).all(1)):
        V_net_per.append([[x_max, y_max],[]])

    vert   = np.array([v[0] for v in V_net_per])
    vert_s = np.array([v for v in vert if np.abs(v[0]- x_max)<1e-8])
    vert_s = np.array(sorted(vert_s, key=lambda x: x[1]))

    for i, v in enumerate(vert_s[1:]):
        idx_0 = np.where((vert[:,0] == vert_s[i][0]) & (vert[:,1]== vert_s[i][1]))[0][0]
        idx_1 = np.where((vert[:,0] == v[0]) & (vert[:,1]==v[1]))[0][0]
        V_net_per[idx_0][1].append(idx_1)
        V_net_per[idx_1][1].append(idx_0)
    
    if not any(np.equal([x_min,y_max],[v[0] for v in V_net_per]).all(1)):
        V_net_per.append([[x_min, y_max],[]])

    vert = np.array([v[0] for v in V_net_per])
    vert_s = np.array([v for v in vert if np.abs(v[1] - y_max)<1e-8 ])
    vert_s = np.array(sorted(vert_s, key=lambda x: x[0]))

    for i, v in enumerate(vert_s[1:]):
        idx_0 = np.where((vert[:,0] == vert_s[i][0]) & (vert[:,1]== vert_s[i][1]))[0][0]
        idx_1 = np.where((vert[:,0] == v[0]) & (vert[:,1]==v[1]))[0][0]
        V_net_per[idx_0][1].append(idx_1)
        V_net_per[idx_1][1].append(idx_0)


    vert_s = np.array([v for v in vert if np.abs(v[0] - x_min)<1e-8])
    vert_s = np.array(sorted(vert_s, key=lambda x: x[1]))

    for i, v in enumerate(vert_s[1:]):
        idx_0 = np.where((vert[:,0] == vert_s[i][0]) & (vert[:,1]== vert_s[i][1]))[0][0]
        idx_1 = np.where((vert[:,0] == v[0]) & (vert[:,1]==v[1]))[0][0]
        V_net_per[idx_0][1].append(idx_1)
        V_net_per[idx_1][1].append(idx_0)
   
    V_out = [[np.array(v[0]),np.unique(np.array(v[1]))] for v in V_net_per]
    return V_out

def get_boundary_point(vert, atom_0, atom_1, center, x_min, x_max, y_min, y_max, vert_out = None):
    t = atom_1- atom_0
    t = t/np.linalg.norm(t)
    n = np.array([-t[1],t[0]])
    midpoint =np.array([atom_0,atom_1]).mean(axis=0)
    normal = np.sign(np.dot(midpoint-center, n))*n

    inv = False
    if vert_out is not None:
        pt = vert_out - vert
        if np.sign(normal[0]) != np.sign(pt[0]) or np.sign(normal[0]) != np.sign(pt[0]):
            normal[0] = np.sign(pt[0])*np.abs(normal[0])  
            normal[1] = np.sign(pt[1])*np.abs(normal[1])
            inv = True

    if normal[0] < 0 and normal[1] < 0:
        dx = vert[0]-x_min
        dy = vert[1]-y_min
        scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
    elif normal[0] > 0 and normal[1] > 0:
        dx = x_max-vert[0]
        dy = y_max-vert[1]
        scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
    elif normal[0] < 0 and normal[1] > 0:
        dx = vert[0]-x_min
        dy = y_max-vert[1]
        scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
    elif normal[0] > 0 and normal[1] < 0:
        dx = x_max-vert[0]
        dy = vert[1]-y_min
        scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
    elif normal[0] == 0 and normal[1] < 0:
        dy = vert[1]-y_min
        scalar = np.sign(normal[1])*dy/normal[1]
    elif normal[0] == 0 and normal[1] > 0:
        dy = y_max-vert[1]
        scalar = np.sign(normal[1])*dy/normal[1]
    elif normal[0] < 0 and normal[1] == 0:
        dx = vert[0]-x_min
        scalar = np.sign(normal[0])*dx/normal[0]
    elif normal[0] > 0 and normal[1] == 0:
        dx = x_max-vert[0]
        scalar = np.sign(normal[0])*dx/normal[0]
    else:
        scalar = 0
    if inv:
        scalar *= -1
    return np.around(vert + np.sign(np.dot(midpoint-center, n))*n*scalar, decimals=8)

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

            bp = get_boundary_point(vert[i], atoms[pointidx[0]], atoms[pointidx[1]], center, x_min, x_max, y_min, y_max)
           
            vert = np.vstack((vert, bp))
            edge[j][np.argmin(rv)] = len(vert)-1 #only 2D for now

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

def get_cmap(n, name='hsv'):
    '''coloring for visualize'''
    return plt.cm.get_cmap(name, n)

def visualize(mat,coords, sl, v_net, atoms):
    cmap = get_cmap(mat.shape[1]+1)
    for i, col in enumerate(mat.transpose()):
        for k, point in enumerate(col):
            if point and coords[k][2] == sl:
                plt.plot(coords[k][0],coords[k][1], color =cmap(i) , marker='x')
    
    vert = np.array([elem[0] for elem in v_net])
    
    atom = np.array([[a[0],a[1]] for a in atoms])
    plt.plot(atom[:,0], atom[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')
    for v in v_net:
        for c in v[1]:
            plt.plot([v[0][0],v_net[c][0][0]],[v[0][1],v_net[c][0][1]],'k-')
    #plt.xlim(x_min -1.5,x_max +1.5)
    #plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

def get_dist_atom(atoms, dx, dy, point): 
    k_out = -1
    dist = 0
    for k, atom in enumerate(atoms):
        #d = la.norm(atom - point)
        d = per_dist(atom, point, dx, dy)
        if k == 0:
            dist = d
            k_out = k
        elif d < dist:
            dist = d
            k_out = k
    return k_out

def per_dist(atom, point, dx, dy):

    # upper left 
    dist = la.norm(point - [atom[0] -dx ,atom[1]+dy, atom[2]])
    
    # upper mid
    d = la.norm(point - [atom[0],atom[1]+dy, atom[2]])
    if d < dist:
        dist = d

    # upper right
    d = la.norm(point - [atom[0]+dx,atom[1]+dy, atom[2]])
    if d < dist:
        dist = d

    # left 
    d = la.norm(point - [atom[0]-dx,atom[1], atom[2]])
    if d < dist:
        dist = d

    # mid
    d = la.norm(point - [atom[0],atom[1], atom[2]])
    if d < dist:
        dist = d

    # right
    d = la.norm(point - [atom[0]+dx,atom[1], atom[2]])
    if d < dist:
        dist = d

    # lower left 
    d = la.norm(point - [atom[0]-dx,atom[1]-dy, atom[2]])
    if d < dist:
        dist = d

    # lower mid
    d = la.norm(point - [atom[0],atom[1]-dy, atom[2]])
    if d < dist:
        dist = d

    # lower right
    d = la.norm(point - [atom[0]+dx,atom[1]-dy, atom[2]])
    if d < dist:
        dist = d

    return dist


















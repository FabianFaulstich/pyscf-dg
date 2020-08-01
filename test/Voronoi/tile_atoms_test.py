import sys
sys.path.append('../../src')
import dg_tools as dg

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,ConvexHull

# Testing cubical symmetry
def test_cube():
    atoms = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
    x_max =  3
    x_min = -3
    y_max =  3
    y_min = -3

    atoms = dg.tile_atoms(atoms,x_max - x_min, y_max - y_min)
    voronoi  = Voronoi(atoms)

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
    plt.plot([[x_min,x_min],[x_min,x_max],[x_max,x_max],[x_max,x_min]],[[y_min,y_max],[y_max,y_max],[y_max,y_min],[y_min,y_min]],'r-')
    plt.xlim(x_min-(x_max-x_min),x_max+(x_max-x_min))
    plt.ylim(y_min-(y_max-y_min),y_max+(y_max-y_min))
    plt.show()

# Testing distorted cubical symmetry
def test_d_cube():
    atoms = np.array([[1.5,.5],[1.25,-.5],[-1,0.8],[-1,-1]])
    x_max =  3
    x_min = -3
    y_max =  3
    y_min = -3
    
    atoms = dg.tile_atoms(atoms,x_max - x_min, y_max - y_min)
    voronoi  = Voronoi(atoms)

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
    plt.plot([[x_min,x_min],[x_min,x_max],[x_max,x_max],[x_max,x_min]],[[y_min,y_max],[y_max,y_max],[y_max,y_min],[y_min,y_min]],'r-')
    plt.xlim(x_min-(x_max-x_min),x_max+(x_max-x_min))
    plt.ylim(y_min-(y_max-y_min),y_max+(y_max-y_min))
    plt.show()

# Testing Ethylene structure
def test_C2H4(detail=False):
    atoms = np.array([[-2.4, -1.86], [-2.4, 1.86], [-1.34, 0], [1.34, 0],[2.4,-1.86], [2.4,1.86]])

    x_max =  6
    x_min = -6
    y_max =  6
    y_min = -6

    atoms = dg.tile_atoms(atoms,x_max - x_min, y_max - y_min)
    voronoi  = Voronoi(atoms)

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
    plt.plot([[x_min,x_min],[x_min,x_max],[x_max,x_max],[x_max,x_min]],[[y_min,y_max],[y_max,y_max],[y_max,y_min],[y_min,y_min]],'r-')
    plt.xlim(x_min-(x_max-x_min),x_max+(x_max-x_min))
    plt.ylim(y_min-(y_max-y_min),y_max+(y_max-y_min))
    plt.show()
    

def test_d_chain_H4(detail = False):

    atoms = np.array([[-3, .3], [-1, -.1],
                      [1, .2], [3, -.4]])

    x_max =  6
    x_min = -6
    y_max =  6
    y_min = -6
    
    atoms = dg.tile_atoms(atoms,x_max - x_min, y_max - y_min)
    voronoi  = Voronoi(atoms)

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
    plt.plot([[x_min,x_min],[x_min,x_max],[x_max,x_max],[x_max,x_min]],[[y_min,y_max],[y_max,y_max],[y_max,y_min],[y_min,y_min]],'r-')
    plt.xlim(x_min-(x_max-x_min),x_max+(x_max-x_min))
    plt.ylim(y_min-(y_max-y_min),y_max+(y_max-y_min))
    plt.show()

def test_chain_H4(detail = False):

    atoms = np.array([[-3, 0], [-1, 0],
                      [1, 0], [3, 0]])

    x_max =  6
    x_min = -6
    y_max =  6
    y_min = -6

    atoms = dg.tile_atoms(atoms,x_max - x_min, y_max - y_min)
    voronoi  = Voronoi(atoms)

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
    plt.plot([[x_min,x_min],[x_min,x_max],[x_max,x_max],[x_max,x_min]],[[y_min,y_max],[y_max,y_max],[y_max,y_min],[y_min,y_min]],'r-')
    plt.xlim(x_min-(x_max-x_min),x_max+(x_max-x_min))
    plt.ylim(y_min-(y_max-y_min),y_max+(y_max-y_min))
    plt.show()

def test_chain_H2(detail = False):


    atoms = np.array([[5.4, 3], [6.6, 3]])

    x_max = 12
    x_min = 0
    y_max = 6
    y_min = 0

    atoms = dg.tile_atoms(atoms,x_max - x_min, y_max - y_min)
    voronoi  = Voronoi(atoms)
  
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
    plt.plot([[x_min,x_min],[x_min,x_max],[x_max,x_max],[x_max,x_min]],[[y_min,y_max],[y_max,y_max],[y_max,y_min],[y_min,y_min]],'r-')
    plt.xlim(x_min-(x_max-x_min),x_max+(x_max-x_min))
    plt.ylim(y_min-(y_max-y_min),y_max+(y_max-y_min))
    plt.show()


if __name__ == '__main__':

    # Testing H2
    test_chain_H2()
    
    # Testing quasi 1D systems
    test_chain_H4()

    # Testing distorted quasi 1D systems
    test_d_chain_H4()

    # Testing cubical symmetry 
    test_cube()

    # Testing distorted cubical symmetry
    test_d_cube()

    # Testing Ethylene symmetry
    test_C2H4()


import sys
sys.path.append('../../src')
import dg_tools as dg

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

from scipy.spatial import ConvexHull

def mol_size(Mol):

    expan = lambda cords: max(cords) - min(cords)
    minim = lambda cords: min(cords)
    out   = []
    m     =[]

    for i in range(3):
        out.append(expan([mol[i] for mol in Mol]))
        m.append(minim([mol[i] for mol in Mol]))

    return np.array(out), m

def RandomAtoms():
    xx = np.linspace(0, 5, 11)
    yy = np.linspace(0, 5, 11)
    zz = np.linspace(0, 5, 11)

    grid    = np.array([[x,y,z] for x in xx for y in yy for z in zz])
    atoms   = np.random.randint(0,5,(6,3)) 
    #np.array([[1,1,1], [4,4,1]])#, [4,1,1]])
    #print(atoms)
    #exit()

    start = time.time()
    print("Starting voronoi ...")
    idx_mat = dg.naive_voronoi(grid, atoms, 5, 5, 5)
    print('Done! Elapsed time: ', time.time() - start)
    
    cmap = dg.get_cmap(len(atoms) + 1)
    fig = plt.figure(figsize = (10, 7)) 
    ax = plt.axes(projection ="3d") 

    for atom in atoms:
        ax.scatter3D(atom[0], atom[1], atom[2], color = "k", marker = "D");
    for i, col in enumerate(idx_mat.transpose()):
        for k, point in enumerate(col):
            if point:
                ax.scatter3D(grid[k][0], grid[k][1], grid[k][2], 
                        color = cmap(i), marker = "+");            
    plt.show()


def CH4():
    xx = np.linspace(0, 18, 91)
    yy = np.linspace(0, 18, 91)
    zz = np.linspace(0, 18, 91)

    grid    = np.array([[x,y,z] for x in xx for y in yy for z in zz])

    atoms = [[ 0, 0, 0], [ 1.186, 1.186, 1.186],
             [ 1.186,-1.186,-1.186],[-1.186, 1.186,-1.186],
             [-1.186,-1.186, 1.186]]

    # Centering Molecule in Box:
    ms, mm = mol_size(atoms)
    offset = np.array([ (bs - s)/2. - m for bs, s, m in zip([18]*3, ms, mm)])

    for k, off in enumerate(offset):
        for j in range(len(atoms)):
            atoms[j][k] += off

    
    start = time.time()
    print("Starting voronoi ...")
    idx_mat = dg.naive_voronoi(grid, atoms, 18, 18, 18)
    print('Done! Elapsed time: ', time.time() - start)

    cmap = dg.get_cmap(len(atoms) + 1)
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    #for i, col in enumerate(idx_mat.transpose()):
    #    if i == 0:
    #        for k, point in enumerate(col):
    #            if point:
    #                ax.scatter3D(grid[k][0], grid[k][1], grid[k][2],
    #                        cmap='viridis', marker = ".");
    
    #vcell = idx_mat[:,1]
    #points = grid[vcell,:]
    #ax.scatter3D(points[:,0], points[:,1], points[:,2], 
    #        cmap='green', edgecolor='r'); 
    vcell = idx_mat[:,0]
    points = grid[vcell,:]
    hullpoints = ConvexHull(points)
    for simplex in hullpoints.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')

    #ax.plot_trisurf(hullpoints[:,0], hullpoints[:,1], hullpoints[:,2], 
    #        cmap='viridis', edgecolor='None');
    ax.scatter3D(points[:,0], points[:,1], points[:,2], 
            cmap='red', edgecolor='None');
    for i, atom in enumerate(atoms):
        if i == 0:
            ax.plot([atom[0]], [atom[1]], [atom[2]], color = "k", marker = ".", 
                    markersize=50)
        else:
            ax.plot([atom[0],atoms[0][0]],[atom[1],atoms[0][1]],
                    [atom[2],atoms[0][2]], color = 'k', linewidth=4)
            ax.plot([atom[0]], [atom[1]], [atom[2]], color = "b", marker = ".",
                    markersize=40)
            
    plt.show()

def timing():

    xx = np.linspace(0, 12, 25)
    yy = np.linspace(0, 12, 25)
    zz = np.linspace(0, 12, 25)

    grid    = np.array([[x,y,z] for x in xx for y in yy for z in zz])
    atoms   = np.array([[5, 5, 6], [7, 5, 6], [5, 7, 6], [7, 7, 6]]) 
      
    
    print("Starting naive Voronoi:")
    vstart = time.time()
    idx_mat = np.zeros((grid.shape[0],len(atoms)), dtype = bool)
    for j, point in enumerate(grid):
        k = dg.get_dist_atom(atoms, 12, 12, point)
        idx_mat[j,k] = True
    vend = time.time()
    print(idx_mat.shape)
    print("Computational time for naive voronoi: ", vend - vstart)
    
    start = time.time()
    print("Starting voronoi ...")
    idx_mat = dg.naive_voronoi(grid, atoms, 12, 12, 12)
    print('Done! Elapsed time: ', time.time() - start)




if __name__ == '__main__':

    CH4()   

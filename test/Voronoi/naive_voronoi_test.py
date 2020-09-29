import sys
sys.path.append('../../src')
import dg_tools as dg

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time


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

    timing()   

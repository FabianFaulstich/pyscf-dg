import sys
sys.path.append('..')
import dg_model_ham as dg

import numpy as np
import numpy.matlib
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, Delaunay
import time
from scipy.optimize import linprog

import pyscf
from pyscf import lib
from pyscf.pbc import dft as dft
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import FFTDF
from pyscf.pbc import tools
from pyscf import ao2mo

def in_hull(p, hull):
    """
    Test if points in p are in hull
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


if __name__ == '__main__':
    # an example of C2H4 molecul
    print(pyscf.__version__)
    
    Mol = [['H',[-2.4, -1.86, 0]], ['H',[-2.4, 1.86, 0]], ['C',[-1.34, 0, 0]], ['C',[1.34, 0, 0]],
                       ['H',[2.4,-1.86, 0]], ['H',[2.4,1.86,0 ]]]

    atom_pos = np.array([atom[1] for atom in Mol])
    atoms    = [atom[0] for atom in Mol]
    Mol_size = np.array([np.amax(atom_pos[:,i])-np.amin(atom_pos[:,i]) for i in range(3)])
    offset   = np.array([6,6,6])
    atom_off = np.array([offset[d]/2.0 - np.amin(atom_pos[:,d]) for d in range(3)])

    X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
    atom_box = np.array([pos + atom_off for pos in atom_pos])
    Mol_box  = [[atoms[i],atom_box[i]] for i in range(len(atoms))] 
    
    print(atom_box)

    dgrid = [4]*3
    
    cell = gto.Cell()
    cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
    cell.unit    = 'B'
    cell.verbose = 3
    cell.basis   = 'sto-3g' #gth-dzvp
    cell.pseudo  = 'gth-pade'
    # potential speed up 
    #cell.ke_cutoff = 100.0
    cell.mesh    = np.array([int(d * x) for d, x in zip(dgrid, X)])
    cell.atom    = Mol_box
    cell.build()

    # HF
    #print("Computing HF in " + cell.basis +  " basis ...")
    #start_hf = time.time()
    #mf = scf.RHF(cell, exxdiv='None') # madelung correction: ewlad
    #mf.kernel()
    #mfe[i] = mf.e_tot
    #end_hf   = time.time()
    #print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
    #print()

    # Voronoi 3D:
    #vor_3d = Voronoi(atom_box)
    #print("3D Voronoi RV's:")
    #print(vor_3d.ridge_vertices)


    atoms_2d = np.array([atom[1][:2] for atom in Mol_box])
    mesh_2d  = np.unique(cell.get_uniform_grids()[:,:2], axis=0)
    voronoi  = Voronoi(atoms_2d)

    ''' vertices: list of lists
        vertices of the Voronoi decomp.
        ridge_vertices: list of lists
        Connected vertices
    '''

    print("Voronoi vertices:")
    print(voronoi.vertices)
    print("Voronoi ridge_vertices:")
    print(voronoi.ridge_vertices)
        
    print("Determining voronoi cells:")
    
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([5.4, np.amin(mesh_2d[:,1])])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([5.4, np.amax(mesh_2d[:,1])])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amin(mesh_2d[:,0]), 4.86])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amax(mesh_2d[:,0]), 4.86])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amin(mesh_2d[:,0]), np.amin(mesh_2d[:,1])])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amax(mesh_2d[:,0]), np.amax(mesh_2d[:,1])])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amin(mesh_2d[:,0]), np.amax(mesh_2d[:,1])])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amax(mesh_2d[:,0]), np.amin(mesh_2d[:,1])]))) 
    vert = voronoi.vertices
    
    print(vert)
    voronoi_cells = []
    voronoi_cells.append(np.array([vert[8], vert[4], vert[2], vert[0], vert[6]]))
    voronoi_cells.append(np.array([vert[6], vert[0], vert[3], vert[5], vert[10]]))
    voronoi_cells.append(np.array([vert[2], vert[3], vert[0]]))
    voronoi_cells.append(np.array([vert[3], vert[1], vert[7], vert[9], vert[5]]))
    voronoi_cells.append(np.array([vert[2], vert[1], vert[3]]))
    voronoi_cells.append(np.array([vert[4], vert[11], vert[7], vert[1], vert[2]]))
    print(voronoi_cells)

    for vcell in voronoi_cells:
        hull = ConvexHull(vcell)
        for simplex in hull.simplices:
            plt.plot(vcell[simplex, 0],vcell[simplex, 1], 'k-')
    
    color_code = ['gx','bx','mx','yx','kx','rx',]
    for k, vcell in enumerate(voronoi_cells):
        mesh = in_hull(mesh_2d, vcell)
        for i, point in enumerate(mesh_2d):
            if mesh[i]:
                plt.plot(point[0], point[1], color_code[k])


    # plotting conections between two voronoi vertices: 
    #for rv in voronoi.ridge_vertices:
    #    rv = np.asarray(rv)
    #    if np.all(rv >= 0):
    #        plt.plot(voronoi.vertices[rv,0], voronoi.vertices[rv,1], 'k-')
    
    # plotting connextions from box limit with voronoi 
    #center = atoms_2d.mean(axis=0)
    #plt.plot(center[0],center[1],'r+')
    #print("center: ",center)
    #print("Ridge points:")
    #print(voronoi.ridge_points)
    #for pointidx, rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
    #    rv = np.asarray(rv)
    #    if np.any(rv < 0):
    #        i = rv[rv >= 0][0]
    #        t = atoms_2d[pointidx[1]]- atoms_2d[pointidx[0]]
    #        t = t/np.linalg.norm(t)
    #        n = np.array([-t[1],t[0]])
    #        midpoint = atoms_2d[pointidx].mean(axis=0)
    #        far_point = voronoi.vertices[i] + np.sign(np.dot(midpoint-center, n)) * n * 100
    #        plt.plot([voronoi.vertices[i,0],far_point[0]], [voronoi.vertices[i,1], far_point[1]], 'k-')

    # plotting voronoi vertices: 
    plt.plot(voronoi.vertices[:,0], voronoi.vertices[:,1],'D')  # poltting voronoi vertices

    plt.plot(mesh_2d[:,0], mesh_2d[:,1], ',')
    plt.plot(atoms_2d[:,0], atoms_2d[:,1], 'bo')
    plt.xlim(np.amin(mesh_2d[:,0]) -1.5,np.amax(mesh_2d[:,0]) +1.5)
    plt.ylim(np.amin(mesh_2d[:,1]) -1.5,np.amax(mesh_2d[:,1]) +1.5)

    plt.show()

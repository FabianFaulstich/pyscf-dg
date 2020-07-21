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
from pyscf import gto as gto_mol
from pyscf import scf as scf_mol
from pyscf import cc  as cc_mol

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
    # an example of H4 molecul
    print(pyscf.__version__)
   
    Mol = [['H',[1, 1, 0]], ['H',[2, -1, 0]], ['H',[-.5, -1.5, 0]], ['H',[-1, .25, 0]]]

    atom_pos = np.array([atom[1] for atom in Mol])
    atoms    = [atom[0] for atom in Mol]
    Mol_size = np.array([np.amax(atom_pos[:,i])-np.amin(atom_pos[:,i]) for i in range(3)])
    offset   = np.array([3,3,3])
    atom_off = np.array([offset[d]/2.0 - np.amin(atom_pos[:,d]) for d in range(3)])

    X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
    atom_box = np.array([pos + atom_off for pos in atom_pos])
    Mol_box  = [[atoms[i],atom_box[i]] for i in range(len(atoms))] 
    
    dgrid = [6]*3
    
    cell = gto.Cell()
    cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
    cell.unit    = 'B'
    cell.verbose = 3
    cell.basis   = 'sto-3g' #gth-dzvp
    cell.pseudo  = 'gth-pade'
    # potential speed up 
    cell.ke_cutoff = 200.0
    cell.mesh    = np.array([int(d * x) for d, x in zip(dgrid, X)])
    cell.atom    = Mol_box
    cell.build()
   
    mol = gto_mol.M()
    mol.basis = 'sto-3g'
    mol.atom = Mol_box
    mol.unit = 'bohr'
    mol.build()

    mf   = scf_mol.RHF(mol).run()
    mfe  = mf.e_tot
    mycc = cc_mol.CCSD(mf)
    mycc.kernel()
    ecc  = mycc.e_corr

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
    
    # Determining voronoi cells
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amin(mesh_2d[:,0]), np.amin(mesh_2d[:,1])])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amax(mesh_2d[:,0]), np.amax(mesh_2d[:,1])])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amin(mesh_2d[:,0]), np.amax(mesh_2d[:,1])])))
    voronoi.vertices = np.vstack((voronoi.vertices, np.array([np.amax(mesh_2d[:,0]), np.amin(mesh_2d[:,1])]))) 
    
    # Compute vertices on the boundary
    center = atoms_2d.mean(axis=0)
    for pointidx, rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        rv = np.asarray(rv)
        if np.any(rv < 0):
            i = rv[rv >= 0][0]
            t = atoms_2d[pointidx[1]]- atoms_2d[pointidx[0]]
            t = t/np.linalg.norm(t)
            n = np.array([-t[1],t[0]])
            midpoint = atoms_2d[pointidx].mean(axis=0)
            normal = np.sign(np.dot(midpoint-center, n))*n 
            if normal[0] < 0 and normal[1] < 0:
                dx = voronoi.vertices[i][0]-np.amin(mesh_2d[:,0])
                dy = voronoi.vertices[i][1]-np.amin(mesh_2d[:,1])
                scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )    
            elif normal[0] > 0 and normal[1] > 0:
                dx = np.amax(mesh_2d[:,0])-voronoi.vertices[i][0]
                dy = np.amax(mesh_2d[:,1])-voronoi.vertices[i][1]
                scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
            elif normal[0] < 0 and normal[1] > 0:
                dx = voronoi.vertices[i][0]-np.amin(mesh_2d[:,0])
                dy = np.amax(mesh_2d[:,1])-voronoi.vertices[i][1]
                scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
            elif normal[0] > 0 and normal[1] < 0:
                dx = np.amax(mesh_2d[:,0])-voronoi.vertices[i][0]
                dy = voronoi.vertices[i][1]-np.amin(mesh_2d[:,1])
                scalar = min(np.sign(normal[0])*dx/normal[0] ,np.sign(normal[1])*dy/normal[1] )
            #elif one of them is zero!!!
            far_point = voronoi.vertices[i] + np.sign(np.dot(midpoint-center, n))*n*scalar 
            voronoi.vertices = np.vstack((voronoi.vertices, far_point))
    
    vert = voronoi.vertices

    voronoi_cells = []
    voronoi_cells.append(np.array([vert[2], vert[7], vert[1], vert[0], vert[6]]))
    voronoi_cells.append(np.array([vert[6], vert[0], vert[8], vert[4]]))
    voronoi_cells.append(np.array([vert[8], vert[0], vert[1], vert[9], vert[3]]))
    voronoi_cells.append(np.array([vert[1], vert[9], vert[5], vert[7]]))

    # DG vs VDG calculations
    print("Creating  " + cell.basis +  "-VDG Hamiltonian ...")
    start_dg = time.time()
    cell_vdg  = dg.dg_model_ham(cell, None ,'rel_num', 0.85, True, voronoi_cells)
    end_dg   = time.time()
    print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
    print()

    print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
    start_dg = time.time()
    cell_dg  = dg.dg_model_ham(cell, None ,'rel_num', 0.85)
    end_dg   = time.time()
    print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
    print()

    # HF
    print("Computing HF in " + cell.basis +  "-VDG basis ...")
    start_hf = time.time()
    mfe_vdg = cell_vdg.run_RHF()
    end_hf   = time.time()
    print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
    print()

    print("Computing HF in " + cell.basis +  "-DG basis ...")
    start_hf = time.time()
    mfe_dg = cell_dg.run_RHF()
    end_hf   = time.time()
    print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
    print()

    # MP2
    print("Computing MP2 in " + cell.basis +  "-VDG basis ...")
    start_mp = time.time()
    emp_vdg, _ = cell_vdg.run_MP2()
    end_mp   = time.time()
    print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
    print()

    print("Computing MP2 in " + cell.basis +  "-DG basis ...")
    start_mp = time.time()
    emp_dg, _ = cell_dg.run_MP2()
    end_mp   = time.time()
    print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
    print()

    # CCSD
    print("Computing CCSD in " + cell.basis +  "-VDG basis ...")
    start_cc = time.time()
    ecc_vdg, _ = cell_vdg.run_CC()
    end_cc   = time.time()
    print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
    print()

    print("Computing CCSD in " + cell.basis +  "-DG basis ...")
    start_cc = time.time()
    ecc_dg, _ = cell_dg.run_CC()
    end_cc   = time.time()
    print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
    print()

    print("Meanfield results:")
    print("Mol: ", mfe)
    print("DG: " , mfe_dg)
    print("VDG: ", mfe_vdg)
    print()
    print("MP2 correlation energy:")
    print("DG: " ,emp_dg)
    print("VDG: ",emp_vdg)
    print()
    print("CCSD correlation energy:")
    print("Mol: ", ecc)
    print("DG: " ,ecc_dg)
    print("VDG: ",ecc_vdg)


    for vcell in voronoi_cells:
        hull = ConvexHull(vcell)
        for simplex in hull.simplices:
            plt.plot(vcell[simplex, 0],vcell[simplex, 1], 'k-')
    
    color_code = ['gx','bx','mx','yx']
    for k, vcell in enumerate(voronoi_cells):
        mesh = in_hull(mesh_2d, vcell)
        for i, point in enumerate(mesh_2d):
            if mesh[i]:
                plt.plot(point[0], point[1], color_code[k])

    plt.plot(mesh_2d[:,0], mesh_2d[:,1], ',')
    plt.plot(atoms_2d[:,0], atoms_2d[:,1], 'bo')
    plt.plot(voronoi.vertices[:,0], voronoi.vertices[:,1],'ro')  # poltting voronoi vertices
 
    plt.xlim(np.amin(mesh_2d[:,0]) -1.5,np.amax(mesh_2d[:,0]) +1.5)
    plt.ylim(np.amin(mesh_2d[:,1]) -1.5,np.amax(mesh_2d[:,1]) +1.5)
    plt.show()

import sys
sys.path.append('../../src')

import matplotlib
import matplotlib.pyplot as plt
import dg_model_ham as dg
import numpy as np
from numpy import linalg as la
from scipy.linalg import block_diag
import scipy
import dg_tools

import pyscf
from pyscf import lib
from pyscf.pbc import dft as dft
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import FFTDF
from pyscf.pbc import tools
from pyscf import ao2mo
from sys import exit
import time


if __name__ == '__main__':

    ''' Computing Benzene (C6H6) in x-y plane:
    '''

    start_pes = time.time()

    Mol =[['C', [ 0.    , 1.3915, 0]],
          ['H', [ 0.    , 2.4715, 0]],
          ['C', [ 1.2051, 0.6958, 0]],
          ['H', [ 2.1404, 1.2358, 0]],
          ['C', [ 1.2051,-0.6958, 0]],
          ['H', [ 2.1404,-1.2358, 0]],
          ['C', [ 0.    ,-1.3915, 0]],
          ['H', [ 0.    ,-2.4715, 0]],
          ['C', [-1.2051,-0.6958, 0]],
          ['H', [-2.1404,-1.2358, 0]],
          ['C', [-1.2051, 0.6958, 0]],
          ['H', [-2.1404, 1.2357, 0]]
          ]

     # Placing Molecule in the box:
    atom_pos = np.array([atom[1] for atom in Mol])
    atoms    = [atom[0] for atom in Mol]
    Mol_size = np.array([np.amax(atom_pos[:,i])-np.amin(atom_pos[:,i]) for i in range(3)])
    offset   = np.array([6,6,3]) #                                           
    atom_off = np.array([offset[d]/2.0 - np.amin(atom_pos[:,d]) for d in range(3)])

    X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
    atoms_box = np.array([pos + atom_off for pos in atom_pos])
    Mol_box  = [[atoms[i],atoms_box[i]] for i in range(len(atoms))]

    dgrid = [4]*3

    cell = gto.Cell()
    cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
    cell.unit    = 'B'
    cell.verbose = 3
    cell.basis   = 'gth-dzvp' #gth-dzvp
    cell.pseudo  = 'gth-pade'
    # potential speed up 
    cell.ke_cutoff = 200.0
    cell.mesh    = np.array([int(d * x) for d, x in zip(dgrid, X)])
    cell.atom    = Mol_box
    cell.build()

    # Voronoi 2D:

    # 2D projection of atom position and grid:
    atoms_2d = np.array([atom[1][:2] for atom in Mol_box])
    mesh_2d  = np.unique(cell.get_uniform_grids()[:,:2], axis=0)

    # get Voronoi vertices + vertices on the boundary box (periodic voronoi):
    V_net = dg_tools.get_V_net_per(atoms_2d, np.amin(mesh_2d[:,0]), np.amax(mesh_2d[:,0]),
                        np.amin(mesh_2d[:,1]), np.amax(mesh_2d[:,1]))


    voronoi_cells = dg_tools.get_V_cells(V_net, atoms_2d)
    vert = np.array([elem[0] for elem in V_net])
    # get Voronoi cells:
    
    # DG vs VDG calculations
    print("Creating  " + cell.basis +  "-VDG Hamiltonian ...")
    start_dg = time.time()
    cell_vdg  = dg.dg_model_ham(cell, None ,'rel_num', 0.8, True, voronoi_cells)
    end_dg   = time.time()
    print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
    print()

    print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
    start_dg = time.time()
    cell_dg  = dg.dg_model_ham(cell, None ,'rel_num', 0.8)
    end_dg   = time.time()
    print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
    print()


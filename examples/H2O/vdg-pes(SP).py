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

    ''' Computing H2O in x-y plane:
        bon angle remains fixed,
        O-H bonds get dissociated
    '''
    start_pes = time.time()


    #offsets = np.array([12, 14, 16, 18, 20])
    offsets = np.array([8])
    mfe     = np.zeros(len(offsets))
    mfe_ew  = np.zeros(len(offsets))
    mfe_vdg = np.zeros(len(offsets))
    mfe_ndg = np.zeros(len(offsets))
    
    mf_mad     = np.zeros(len(offsets))
    mf_vdg_mad = np.zeros(len(offsets))
    mf_ndg_mad = np.zeros(len(offsets))

    mf_tes     = np.zeros(len(offsets))
    mf_vdg_tes = np.zeros(len(offsets))
    mf_ndg_tes = np.zeros(len(offsets))

    for k, off in enumerate(offsets):

        # Optimal geometry taken from cccbdb 
        Mol = [['H',[1.43, -.89, 0]], ['O',[0,0,0]], ['H',[-1.43,-.89,0 ]]]

        # Placing Molecule in the box:
        atom_pos = np.array([atom[1] for atom in Mol])
        atoms    = [atom[0] for atom in Mol]
        Mol_size = np.array([np.amax(atom_pos[:,i])-np.amin(atom_pos[:,i]) for i in range(3)])
        offset   = np.array([off,off,off]) #
        atom_off = np.array([offset[d]/2.0 - np.amin(atom_pos[:,d]) for d in range(3)])
        
        X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
        atoms_box = np.array([pos + atom_off for pos in atom_pos])
        Mol_box  = [[atoms[i],atoms_box[i]] for i in range(len(atoms))]
        print(Mol_box)
        
        dgrid = [6]*3
        
        cell = gto.Cell()
        cell.atom    = Mol_box
        cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
        cell.unit    = 'B'
        cell.verbose = 3
        
        cell.basis   = 'gth-dzvp' #gth-dzvp
        cell.pseudo  = 'gth-pade' #reduces the number of electrons???
        # potential speed up 
        #cell.ke_cutoff = 40.0
        cell.mesh    = np.array([int(d * x) for d, x in zip(dgrid, X)])
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

        # DG vs VDG vs None calculations
        print("Creating  " + cell.basis +  "-None-DG Hamiltonian ...")
        start_dg = time.time()
        cell_ndg  = dg.dg_model_ham(cell, None ,'rel_num', 0.95, True, voronoi_cells, V_net, False)
        end_dg   = time.time()
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()
        
        print("Creating  " + cell.basis +  "-VDG Hamiltonian ...")
        start_dg = time.time()
        cell_vdg  = dg.dg_model_ham(cell, None ,'rel_num', 0.95, True, voronoi_cells, V_net)
        end_dg   = time.time()
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()

        print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
        start_dg = time.time()
        #cell_dg  = dg.dg_model_ham(cell, None ,'rel_num', 0.95)
        end_dg   = time.time()
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()
        
        # HF
        print("Computing HF in " + cell.basis +  " basis ...")
        start_hf = time.time()
        mf = scf.RHF(cell, exxdiv='None') # madelung correction: ewlad
        mf.kernel(dump_chk=False)
        mfe[k] = mf.e_tot
        print("HF-Energy (None): ",mfe)
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

        print("Computing HF in " + cell.basis +  " basis ...")
        start_hf = time.time()
        cell.verbose = 3
        mf_ew = scf.RHF(cell, exxdiv='ewald') # madelung correction: ewlad
        mf_mad[k] = tools.pbc.madelung(cell,[mf.kpt])
        mf_tes[k] = mf_mad[k] * cell.nelectron * -0.5
        print("cell atoms: ", cell.atom)
        print("No. of electrons in cell:", cell.nelectron)
        print("Madelung const:", mf_mad[k])
        print("Total enegy shift:", mf_tes[k])
        mf_ew.kernel(dump_chk=False)
        mfe_ew[k] = mf_ew.e_tot
        print("HF-Energy (Ewald): ",mfe_ew)
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()
        
        print("Computing HF in " + cell.basis +  "-None-DG basis ...")
        start_hf = time.time()
        mfe_ndg[k] = cell_ndg.run_RHF(None)
        mf_ndg_mad[k] = cell_ndg.madelung
        mf_ndg_tes[k] = cell_ndg.tes
        print("Madelung const:", mf_ndg_mad[k])
        print("Total enegy shift:", mf_ndg_tes[k])
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()


        print("Computing HF in " + cell.basis +  "-None-DG basis ...")
        start_hf = time.time()
        mfe_ndg[k] = cell_ndg.run_RHF()
        mf_ndg_mad[k] = cell_ndg.madelung
        mf_ndg_tes[k] = cell_ndg.tes
        print("Madelung const:", mf_ndg_mad[k])
        print("Total enegy shift:", mf_ndg_tes[k])
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

        print("Computing HF in " + cell.basis +  "-VDG basis ...")
        start_hf = time.time()
        mfe_vdg[k] = cell_vdg.run_RHF()
        mf_vdg_mad[k] = cell_vdg.madelung
        mf_vdg_tes[k] = cell_vdg.tes
        print("Madelung const:", mf_vdg_mad[k])
        print("Total enegy shift:", mf_vdg_tes[k])
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

    
    
    #print("Computing HF in " + cell.basis +  "-DG basis ...")
    #start_hf = time.time()
    #mfe_dg = cell_dg.run_RHF()
    #end_hf   = time.time()
    #print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
    #print()
    
    # DFT
    #print("Computing HF in " + cell.basis +  "-VDG basis ...")
    #start_hf = time.time()
    #mfe_vdg = cell_vdg.run_RHF()
    #end_hf   = time.time()
    #print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
    #print()

    #print("Computing HF in " + cell.basis +  "-DG basis ...")
    #start_hf = time.time()
    #mfe_dg = cell_dg.run_RHF()
    #end_hf   = time.time()
    #print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
    #print()

    #print("Computing HF in " + cell.basis +  " basis ...")
    #start_hf = time.time()
    #mf = scf.RHF(cell, exxdiv='None') # madelung correction: ewlad
    #mf.kernel(dump_chk=False)
    #mfe = mf.e_tot
    #end_hf   = time.time()
    #print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
    #print()


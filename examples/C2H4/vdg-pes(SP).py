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
    
    ''' Computing C2H4 in x-y plane:
        H-C-H bonds remain relatively fixed,
        C=C bond (in X-direction) gets dissociated
    '''
    
    start_pes = time.time()

    # Optimal geometry taken from cccbdb 
    Mol = [['H',[-2.4, -1.86, 0]], ['H',[-2.4, 1.86, 0]], ['C',[-1.34, 0, 0]], 
           ['C',[1.34, 0, 0]], ['H',[2.4,-1.86, 0]], ['H',[2.4,1.86,0 ]]]
    
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
    cell_vdg  = dg.dg_model_ham(cell, None ,'rel_num', 0.9, True, voronoi_cells)
    end_dg   = time.time()
    print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
    print()

    print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
    start_dg = time.time()
    cell_dg  = dg.dg_model_ham(cell, None ,'rel_num', 0.9)
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
    #emp_vdg, _ = cell_vdg.run_MP2()
    end_mp   = time.time()
    print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
    print()

    print("Computing MP2 in " + cell.basis +  "-DG basis ...")
    start_mp = time.time()
    #emp_dg, _ = cell_dg.run_MP2()
    end_mp   = time.time()
    print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
    print()

    # CCSD
    print("Computing CCSD in " + cell.basis +  "-VDG basis ...")
    start_cc = time.time()
    #ecc_vdg, _ = cell_vdg.run_CC()
    end_cc   = time.time()
    print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
    print()

    print("Computing CCSD in " + cell.basis +  "-DG basis ...")
    start_cc = time.time()
    #ecc_dg, _ = cell_dg.run_CC()
    end_cc   = time.time()
    print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
    print()

    # HF
    print("Computing HF in " + cell.basis +  " basis ...")
    start_hf = time.time()
    mf = scf.RHF(cell, exxdiv='None') # madelung correction: ewlad
    mf.kernel(dump_chk=False)
    mfe = mf.e_tot
    end_hf   = time.time()
    print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
    print()

    # MP2
    print("Computing MP2 in " + cell.basis +  " basis ...")
    start_mp = time.time()
    #emp, _ = mp.MP2(mf).kernel()
    end_mp   = time.time()
    print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
    print()

    # CCSD
    print("Computing CCSD in " + cell.basis +  " basis ...")
    start_cc = time.time()
    #cc_builtin = cc.CCSD(mf)
    #cc_builtin.kernel()
    #ecc = cc_builtin.e_corr
    end_cc   = time.time()
    print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
    print()

    print("Meanfield results:")
    print(cell.basis+ ": ", mfe)
    print(cell.basis+ "-DG: " , mfe_dg)
    print(cell.basis+ "-VDG: ", mfe_vdg)
    print()
    #print("MP2 correlation energy:")
    #print(cell.basis+": ", emp)
    #print(cell.basis+ "-DG: " , emp_dg)
    #print(cell.basis+ "-VDG: ", emp_vdg)
    #print()
    #print("CCSD correlation energy:")
    #print(cell.basis+ ": ", ecc)
    #print(cell.basis+ "-DG: " , ecc_dg)
    #print(cell.basis+ "-VDG: ", ecc_vdg)

    f = open("Output_SP.txt", "w")
    f.write("Meanfield results:\n")
    f.write(cell.basis+ ": "+ str(mfe) + "\n")
    f.write(cell.basis+ "-DG: " + str(mfe_dg)+ "\n")
    f.write(cell.basis+ "-VDG: "+ str(mfe_vdg)+ "\n")
    #f.write("MP2 correlation energy:\n")
    #f.write(cell.basis+": "+ str(emp)+ "\n")
    #f.write(cell.basis+ "-DG: " + str(emp_dg)+ "\n")
    #f.write(cell.basis+ "-VDG: "+ str(emp_vdg)+ "\n")
    #f.write("CCSD correlation energy:\n")
    #f.write(cell.basis+ ": " + str(ecc)+ "\n")
    #f.write(cell.basis+ "-DG: " + str(ecc_dg)+ "\n")
    #f.write(cell.basis+ "-VDG: " + str(ecc_vdg)+ "\n")
    f.close()

import sys
sys.path.append('../../../src')

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

    start_pes = time.time()

    dgrid = [4]*3
    #bonds = np.array([[14e-1,14e-1,14e-1],[16e-1,16e-1,16e-1],[18e-1,18e-1,18e-1],
    #                  [20e-1,20e-1,20e-1],[22e-1,22e-1,22e-1]])
    bonds = np.array([[8e-1,8e-1,8e-1],[10e-1,10e-1,10e-1],[12e-1,12e-1,12e-1],[14e-1,14e-1,14e-1],
                      [16e-1,16e-1,16e-1],[18e-1,18e-1,18e-1],[20e-1,20e-1,20e-1],[24e-1,24e-1,24e-1],
                      [28e-1,28e-1,28e-1], [32e-1,32e-1,32e-1], [36e-1,36e-1,36e-1]])
    bonds_max = np.max(bonds)
    offset = np.array([10., 6., 6.])
    X = np.array([int(np.ceil(offset[0] + bonds_max)), offset[1], offset[2]])

    cell = gto.Cell()
    atom_spec = 'H'
    cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
    cell.unit = 'B'
    cell.verbose = 3
    #cell.basis = '321++g'
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.mesh = np.array([int(d * x) for d, x in zip(dgrid, X)])

    mfe = np.zeros(len(bonds))
    emp = np.zeros(len(bonds))
    ecc = np.zeros(len(bonds))
    
    mfe_dg = np.zeros(len(bonds))
    emp_dg = np.zeros(len(bonds))
    ecc_dg = np.zeros(len(bonds))

    mfe_vdg = np.zeros(len(bonds))
    emp_vdg = np.zeros(len(bonds))
    ecc_vdg = np.zeros(len(bonds))

    for i, bd in enumerate(bonds):
        # atom position in z-direction
        Z = (X[0]-np.sum(bd))/2.0 + np.append( 0, np.cumsum(bd))
        
        cell.atom = [[atom_spec, (Z[0], X[1]/2, X[2]/2)],
                     [atom_spec, (Z[1], X[1]/2, X[2]/2)],
                     [atom_spec, (Z[2], X[1]/2, X[2]/2)],
                     [atom_spec, (Z[3], X[1]/2, X[2]/2)]]

        cell.build()

        # DG vs VDG calculations
        print("Creating  " + cell.basis +  "-VDG Hamiltonian ...")
        start_dg = time.time()

        # Voronoi 2D:
        # 2D projection of atom position and grid:
        atoms_2d = np.array([atom[1][:2] for atom in cell.atom])
        #mesh_2d  = np.unique(cell.get_uniform_grids()[:,:2], axis=0)

        # get Voronoi vertices + vertices on the boundary box (nonperiodic voronoi):
        V_net = dg_tools.get_V_net_per(atoms_2d, 0, cell.a[0][0],0, cell.a[1][1])

        voronoi_cells = dg_tools.get_V_cells(V_net, atoms_2d)

        cell_vdg  = dg.dg_model_ham(cell, None ,'rel_num', 0.99, True, voronoi_cells)
        end_dg   = time.time()
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()
    
        # DG calculations
        print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
        start_dg = time.time()
        cell_dg  = dg.dg_model_ham(cell) 
        end_dg   = time.time() 
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()

        # HF
        print("Computing HF in " + cell.basis +  "-DG basis ...")
        start_hf = time.time()
        mfe_dg[i] = cell_dg.run_RHF()
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()
    
        print("Computing HF in " + cell.basis +  "-VDG basis ...")
        start_hf = time.time()
        mfe_vdg[i] = cell_vdg.run_RHF()
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

        # MP2
        print("Computing MP2 in " + cell.basis +  "-DG basis ...")
        start_mp = time.time()
        emp_dg[i], _ = cell_dg.run_MP2()
        end_mp   = time.time()
        print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
        print()

        print("Computing MP2 in " + cell.basis +  "-VDG basis ...")
        start_mp = time.time()
        emp_vdg[i], _ = cell_vdg.run_MP2()
        end_mp   = time.time()
        print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
        print()

        # CCSD
        print("Computing CCSD in " + cell.basis +  "-DG basis ...")
        start_cc = time.time()    
        ecc_dg[i], _ = cell_dg.run_CC()
        end_cc   = time.time()
        print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
        print()

        print("Computing CCSD in " + cell.basis +  "-VDG basis ...")
        start_cc = time.time()
        ecc_vdg[i], _ = cell_vdg.run_CC()
        end_cc   = time.time()
        print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
        print()


        # Comparing to PySCF:
        print("Comparing to PySCF:")
        # HF
        print("Computing HF in " + cell.basis +  " basis ...")
        start_hf = time.time()
        mf = scf.RHF(cell, exxdiv='ewald') # madelung correction
        mf.kernel()
        mfe[i] = mf.e_tot
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

        # MP2
        print("Computing MP2 in " + cell.basis +  " basis ...")
        start_mp = time.time()
        emp[i], _ = mp.MP2(mf).kernel()
        end_mp   = time.time()
        print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
        print()

        # CCSD
        print("Computing CCSD in " + cell.basis +  " basis ...")
        start_cc = time.time()
        cc_builtin = cc.CCSD(mf)
        cc_builtin.kernel()
        ecc[i] = cc_builtin.e_corr
        end_cc   = time.time()
        print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
        print()

    end_pes = time.time()
    print("Elapsed time: ", end_pes - start_pes, "sec.")

    print("Meanfield results:")
    print("Builtin: ",mfe)
    print("DG: ", mfe_dg)
    print("VDG: ", mfe_vdg)
    print()
    print("MP2 correlation energy:")
    print("Builtin: ",emp)
    print("DG: ",emp_dg)
    print("VDG: ",emp_vdg)
    print()
    print("CCSD correlation energy:")
    print("Builtin: ",ecc)
    print("DG: ",ecc_dg)
    print("VDG: ",ecc_vdg)

    np.savetxt('Energies.txt',(mfe,emp,ecc,mfe_dg,emp_dg,ecc_dg,mfe_vdg,emp_vdg,ecc_vdg))


    bonds_plt = [bd[0] for bd in bonds]
    plt.plot(bonds_plt, mfe   , 'b-v', label =  'HF  (' + cell.basis + ')')
    plt.plot(bonds_plt, mfe_dg, 'r-v', label =  'HF  (' + cell.basis + '-DG)')
    plt.plot(bonds_plt, mfe_vdg, 'g--v', label =  'HF  (' + cell.basis + '-VDG)')

    plt.plot(bonds_plt, mfe + emp      , 'b-^', label =  'MP2  (' + cell.basis + ')')
    plt.plot(bonds_plt, mfe_dg + emp_dg, 'r-^', label =  'MP2  (' + cell.basis + '-DG)')
    plt.plot(bonds_plt, mfe_vdg + emp_vdg, 'g--^', label =  'MP2  (' + cell.basis + '-VDG)')


    plt.plot(bonds_plt, mfe + ecc      , 'b-x', label =  'CCSD  (' + cell.basis + ')')
    plt.plot(bonds_plt, mfe_dg + ecc_dg, 'r-x', label =  'CCSD  (' + cell.basis + '-DG)')
    plt.plot(bonds_plt, mfe_vdg + ecc_vdg, 'g--x', label =  'CCSD  (' + cell.basis + '-VDG)')
    plt.legend()
    plt.show()

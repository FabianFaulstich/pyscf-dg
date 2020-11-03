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

    dgrid = [5]*3
    #bonds = np.array([[12e-1], [14e-1], [16e-1], [18e-1], [20e-1]])
    bonds = np.array([[8e-1], [10e-1], [12e-1], [14e-1], [16e-1], [18e-1], 
                      [20e-1], [24e-1], [28e-1], [32e-1], [36e-1]])
    
    bonds_max = np.max(bonds)
    offset = np.array([5., 5., 5.])
    X = np.array([int(np.ceil(offset[0] + bonds_max)), offset[1], offset[2]])

    cell = gto.Cell()
    atom_spec = 'H'
    cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
    cell.unit = 'B'
    cell.verbose = 3
    cell.basis = 'ccpvdz'
    cell.pseudo = 'gth-pade'
    cell.mesh = np.array([int(d * x) for d, x in zip(dgrid, X)])

    mfe = np.zeros(len(bonds))
    emp = np.zeros(len(bonds))
    ecc = np.zeros(len(bonds))
    
    mfe_dgv = np.zeros(len(bonds))
    emp_dgv = np.zeros(len(bonds))
    ecc_dgv = np.zeros(len(bonds))

    for i, bd in enumerate(bonds):
        
        # atom position in z-direction
        Z = (X[0]-np.sum(bd))/2.0 + np.append( 0, np.cumsum(bd))
        cell.atom = [[atom_spec, (Z[0], X[1]/2, X[2]/2)],
                     [atom_spec, (Z[1], X[1]/2, X[2]/2)]]
        cell.build()

        # DG-V calculations
        print("Creating  DG-V-" + cell.basis +  " Hamiltonian ...")
        start_dg = time.time()
        
        cell_dgv  = dg.dg_model_ham(cell, dg_cuts = None,
                dg_trunc = 'rel_tol', svd_tol = 1e-2, voronoi = True, 
                dg_on=True, gram = None)
                
        end_dg   = time.time()
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()
        
        print("Computing HF in " + cell.basis +  "-VdG basis ...")
        start_hf = time.time()

        mfe_dgv[i] = cell_dgv.run_RHF()
        
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

        print("Computing MP2 in " + cell.basis +  "-VdG basis ...")
        start_mp = time.time()
        
        emp_dgv[i], _ = cell_dgv.run_MP2()
        
        end_mp   = time.time()
        print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
        print()

        print("Computing CCSD in " + cell.basis +  "-VdG basis ...")
        start_cc = time.time()
        
        ecc_dgv[i], _ = cell_dgv.run_CC()
        
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
    print("DG-V: ", mfe_dgv)
    print()
    print("MP2 correlation energy:")
    print("Builtin: ",emp)
    print("DG-V: ",emp_dgv)
    print()
    print("CCSD correlation energy:")
    print("Builtin: ",ecc)
    print("DG-V: ",ecc_dgv)

    np.savetxt('Energies.txt',(mfe, emp, ecc, mfe_dgv, emp_dgv, ecc_dgv))


    bonds_plt = [bd[0] for bd in bonds]
    plt.plot(bonds_plt, mfe, 'b-v', label =  'HF  (' + cell.basis + ')')
    plt.plot(bonds_plt, mfe_dgv, 'g--v', label =  'HF  (DG-V-' + cell.basis + ')')

    plt.plot(bonds_plt, mfe + emp, 'b-^', label =  'MP2  (' + cell.basis + ')')
    plt.plot(bonds_plt, mfe_dgv + emp_dgv, 'g--^', 
            label =  'MP2  (DG-V' + cell.basis + ')')


    plt.plot(bonds_plt, mfe + ecc, 'b-x', label =  'CCSD  (' + cell.basis + ')')
    plt.plot(bonds_plt, mfe_dgv + ecc_dgv, 'g--x', 
            label =  'CCSD  (DG-V-' + cell.basis + ')')
    plt.legend()
    plt.show()


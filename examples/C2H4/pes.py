import sys
sys.path.append('../../src')

import matplotlib
import matplotlib.pyplot as plt
import dg_model_ham as dg
import numpy as np
from numpy import linalg as la
from scipy.linalg import block_diag
import scipy

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
    
    # dissociation in x-direction relative to the optimal geometry
    rel_dis = np.array([8, 10, 12, 14])

    mfe = np.zeros(len(rel_dis))                                                  
    emp = np.zeros(len(rel_dis))                                                  
    ecc = np.zeros(len(rel_dis))                                                  
                                                                                
    mfe_dg = np.zeros(len(rel_dis))                                               
    emp_dg = np.zeros(len(rel_dis))                                               
    ecc_dg = np.zeros(len(rel_dis))

    for i, diss in enumerate(rel_dis): 
        # Placing Molecule in the box:
        atom_pos = np.array([atom[1] for atom in Mol])
        atom_pos[int(len(atom_pos)/2):,0] += diss
        atoms    = [atom[0] for atom in Mol]
        offset    = np.array([8,8,8])                                           
        atom_off  = np.array([offset[d]/2.0 - np.amin(atom_pos[:,d]) for d in range(3)])
        atoms_box = np.array([pos + atom_off for pos in atom_pos]) 
        atoms_box = np.array([pos + atom_off for pos in atom_pos])              
        Mol_box  = [[atoms[i],atoms_box[i]] for i in range(len(atoms))]
        
        # defining largest box needed for this strecht:
        atom_pos_max = np.array([atom[1] for atom in Mol]) 
        atom_pos_max[int(len(atom_pos_max)/2):,0] += np.amax(rel_dis)
        Mol_size = np.array([np.amax(atom_pos_max[:,i])-np.amin(atom_pos_max[:,i]) for i in range(3)])
        X        = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
   
        dgrid = [5]*3

        cell = gto.Cell()
        cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
        cell.unit    = 'bohr'
        cell.verbose = 3
        cell.basis   = 'sto-3g' #gth-dzvp
        cell.pseudo  = 'gth-pade'
        #cell.ke_cutoff = 90000.0
        cell.mesh    = np.array([int(d * x) for d, x in zip(dgrid, X)])
        cell.atom    = Mol_box
        cell.build()

        
        # HF
        #print("Computing HF in " + cell.basis +  " basis ...")
        #start_hf = time.time()
        #mf = scf.RHF(cell, exxdiv='ewald') # madelung correction: ewald
        #mf.kernel()
        #mfe[i] = mf.e_tot
        #end_hf   = time.time()
        #print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        #print()

        
        # DG calculations
        print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
        start_dg = time.time()
        #cell_dg  = dg.dg_model_ham(cell) # default setting dg_trunc = 'abs_tol', svd_tol = 1e-3
        #cell_dg  = dg.dg_model_ham(cell, 'rel_tol', 0.01) 
        #cell_dg  = dg.dg_model_ham(cell, 'abs_num', 4) # set svd_tol > 10 to provoke invalid request
        cell_dg  = dg.dg_model_ham(cell, None ,'rel_num', 0.95)
        #cell_dg  = dg.dg_model_ham(cell, 'abs_cum', 35)
        #cell_dg  = dg.dg_model_ham(cell, 'rel_cum', 2)
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
        
        # MP2
        print("Computing MP2 in " + cell.basis +  "-DG basis ...")
        start_mp = time.time()
        emp_dg[i], _ = cell_dg.run_MP2()
        end_mp   = time.time()
        print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
        print()
        
#        # CCSD
#        print("Computing CCSD in " + cell.basis +  "-DG basis ...")
#        start_cc = time.time()
#        ecc_dg[i], _ = cell_dg.run_CC()
#        end_cc   = time.time()
#        print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
#        print()


#        # Comparing to PySCF:
#        print("Comparing to PySCF:")
#        # HF
#        print("Computing HF in " + cell.basis +  " basis ...")
#        start_hf = time.time()
#        mf = scf.RHF(cell, exxdiv='ewald') # madelung correction
#        mf.kernel()
#        mfe[i] = mf.e_tot
#        end_hf   = time.time()
#        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
#        print()

#        # MP2
#        print("Computing MP2 in " + cell.basis +  " basis ...")
#        start_mp = time.time()
#        emp[i], _ = mp.MP2(mf).kernel()
#        end_mp   = time.time()
#        print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
#        print()

#        # CCSD
#        print("Computing CCSD in " + cell.basis +  " basis ...")
#        start_cc = time.time()
#        cc_builtin = cc.CCSD(mf)
#        cc_builtin.kernel()
#        ecc[i] = cc_builtin.e_corr
#        end_cc   = time.time()
#        print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
#        print()

    end_pes = time.time()
    print("Elapsed time: ", end_pes - start_pes, "sec.")


    print("Meanfield results:")
    print("Builtin: ",mfe)
    print("DG: ", mfe_dg)
    print()
    print("MP2 correlation energy:")
    print("Builtin: ",emp)
    print("DG: ",emp_dg)
    print()
    print("CCSD correlation energy:")
    print("Builtin: ",ecc)
    print("DG: ",ecc_dg)

    np.savetxt('Energies.txt',(mfe,emp,ecc,mfe_dg,emp_dg,ecc_dg))


#    plt.plot(rel_dis + 2.68, mfe   , 'b-v', label =  'HF  (' + cell.basis + ')')
    plt.plot(rel_dis + 2.68, mfe_dg, 'r-v', label =  'HF  (' + cell.basis + '-DG)')

#    plt.plot(rel_dis + 2.68, mfe + emp      , 'b-^', label =  'MP2  (' + cell.basis + ')')
    plt.plot(rel_dis + 2.68, mfe_dg + emp_dg, 'r-^', label =  'MP2  (' + cell.basis + '-DG)')

#    plt.plot(rel_dis + 2.68, mfe + ecc      , 'b-x', label =  'CCSD  (' + cell.basis + ')')
#    plt.plot(rel_dis + 2.68, mfe_dg + ecc_dg, 'r-x', label =  'CCSD  (' + cell.basis + '-DG)')
    plt.legend()
    plt.show()


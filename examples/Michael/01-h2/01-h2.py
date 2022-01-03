import sys
sys.path.append('../../../src')

import matplotlib
import matplotlib.pyplot as plt
import dg_model_ham as dg
import numpy as np
from numpy import linalg as la
from scipy.linalg import block_diag
import scipy
import scipy.io
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
import os

def plot(bonds, mfe, mfe_dgv, emp, emp_dgv, ecc, ecc_dgv, e_nuc):

    fig, ax = plt.subplots(3)

    ax[0].plot(bonds, mfe, label = 'RHF')
    ax[0].plot(bonds, mfe_dgv, label = 'RHF (DG-V)') 
    ax[0].legend()
    ax[0].set_title('Meanfield Energy')

    
    ax[1].plot(bonds, emp, label = 'MP2') 
    ax[1].plot(bonds, emp_dgv, label = 'MP2 (DG-V)') 
    ax[1].plot(bonds, ecc, label = 'CCSD') 
    ax[1].plot(bonds, ecc_dgv, label = 'CCSD (DG-V)') 
    ax[1].legend()
    ax[1].set_title('Correlation Energies')

    ax[2].plot(bonds, mfe_dgv+ ecc_dgv - e_nuc)
    ax[2].set_title('Electronic CCSD energy (DG-V)')
    
    plt.show()


if __name__ == '__main__':

    start_pes = time.time()

    # Discretization of the box (defines the number of PW basis functions)
    dgrid = [5]*3

    # Bond distance of the H2, when computing pes this should be array
    bonds = np.linspace(1, 2.2, num= 13)
    
    # largest bond distance, important for pes, we want to look at the molecule
    # in the same box
    bonds_max = np.amax(bonds)
   
    # pre-defined offset of the molecule to the boudary 
    offset = np.array([5., 5., 5.])
    
    # Defining Box size, atom size + offset
    X = np.array([int(np.ceil(offset[0] + bonds_max)), offset[1], offset[2]])

    # defining Pyscf cell object
    cell = gto.Cell()
    atom_spec = 'H'
    cell.a = [[X[0], 0., 0.], [0., X[1], 0], [0, 0, X[2]]]
    cell.unit = 'B'
    cell.verbose = 3
    cell.basis = '631g'
    cell.pseudo = 'gth-pade'
    cell.mesh = np.array([int(d * x) for d, x in zip(dgrid, X)])

    e_nuc = np.zeros(len(bonds)) 

    # Allocating solution vectors 
    mfe = np.zeros(len(bonds))              
    mfe_dgv = np.zeros(len(bonds))        
                                
    emp = np.zeros(len(bonds))    
    emp_dgv = np.zeros(len(bonds))     
                                
    ecc = np.zeros(len(bonds))    
    ecc_dgv = np.zeros(len(bonds))   


    for m, bd in enumerate(bonds):
        # atom position
        Z = (X[0]-np.sum(bd))/2.0 + np.append( 0, np.cumsum(bd))
        cell.atom = [[atom_spec, (Z[0], X[1]/2, X[2]/2)],
                     [atom_spec, (Z[1], X[1]/2, X[2]/2)]]
        cell.build()

        e_nuc[m] = cell.energy_nuc()

        # Voronoi DG Hamiltonian
        print("Creating  DG-V-" + cell.basis +  " Hamiltonian ...")
        cell_dgv  = dg.dg_model_ham(cell, dg_cuts = None,
                                    dg_trunc = 'abs_num', svd_tol = 10, 
                                    voronoi = True, dg_on=True, gram = None)
        
        print('Done!') 

        vdg_idx = np.array(cell_dgv.dg_idx)
        print('v-dg indices')
        print(vdg_idx)

        h1e_vdg = np.array(cell_dgv.hcore_dg )
        ovl_vdg = np.array(cell_dgv.ovl_dg)
        eri_vdg = np.array(cell_dgv.eri)     

        print('size of h1e: ', h1e_vdg.shape)
        print('size of ovl: ', ovl_vdg.shape)
        print('size of eri: ', eri_vdg.shape)
        print('   non-zero: ', np.count_nonzero(eri_vdg))


        directory = 'output_h2/BD_' + str(bd)
        os.mkdir(directory)
        directory += '/'  
        # Writing to file
        h1e_vdg_mdic = {"h1e_vdg": h1e_vdg}
        scipy.io.savemat(directory+ "h1e_vdg.mat", h1e_vdg_mdic)

        ovl_vdg_mdic = {"ovl_vdg": ovl_vdg}            
        scipy.io.savemat(directory+ "ovl_vdg.mat", ovl_vdg_mdic)  

        # For sanity check
        eri_vdg_mdic = {"eri_vdg": eri_vdg}          
        scipy.io.savemat(directory+ "eri_vdg.mat", eri_vdg_mdic)

        for i in range(len(vdg_idx)):
            name = 'I_'+ str(i)
            idx_vdg_mdic = {name: vdg_idx[i]}            
            scipy.io.savemat(directory+ name+ ".mat", idx_vdg_mdic)  

        for i_1, idx_1 in enumerate(vdg_idx):
            for i_2, idx_2 in enumerate(vdg_idx):
                    sub_eri = eri_vdg[idx_1[:,None, None, None], 
                                      idx_1[:,None, None], 
                                      idx_2[:,None], 
                                      idx_2]
                    print('size of eri subblock: ', sub_eri.shape)
                    name_str = "C_" + str(i_1) + '_' + str(i_2)  
                    sub_eri_mdic = {name_str: sub_eri}
                    scipy.io.savemat(directory+ name_str +".mat", sub_eri_mdic)
       
        # Computing refrence energy:
        # compute ing V-dg basis:

        # RHF
        mfe_dgv[m] = cell_dgv.run_RHF()   
                                          
        # MP2
        emp_dgv[m], _ = cell_dgv.run_MP2()
                                          
        # CCSD
        ecc_dgv[m], _ = cell_dgv.run_CC() 

        
        # Benchmark with PySCF
        # RHF
        mf = scf.RHF(cell, exxdiv='ewald') # madelung correction     
        mf.kernel()                                                  
        mfe[m] = mf.e_tot                                            
                                                                     
        # MP2                                                        
        emp[m], _ = mp.MP2(mf).kernel()                              
                                                                     
        # CCSD                                                       
        cc_builtin = cc.CCSD(mf)                                     
        cc_builtin.kernel()                                          
        ecc[m] = cc_builtin.e_corr                                   

    print(20*"#")
    print("Meanfield results:")           
    print("Builtin: ", mfe)                
    print("DG-V   : ", mfe_dgv)              
    print()                               
    print("MP2 correlation energy:")      
    print("Builtin: ",emp)                
    print("DG-V   : ",emp_dgv)               
    print()                               
    print("CCSD correlation energy:")     
    print("Builtin: ",ecc)                
    print("DG-V   : ",ecc_dgv)               
    print('Nuclear Energy')
    print(e_nuc)
    print(20* '=')
    print('Electronic CCSD energy (DG-V)')
    print(mfe_dgv+ ecc_dgv - e_nuc)

    plot(bonds, mfe, mfe_dgv, emp, emp_dgv, ecc, ecc_dgv, e_nuc)
 

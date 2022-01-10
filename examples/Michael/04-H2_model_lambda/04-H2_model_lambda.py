import sys
sys.path.append('../../../src')

import dg_model_ham as dg
import numpy as np
from numpy import linalg as la
import dg_tools

import matplotlib.pyplot as plt

import scipy
import scipy.io

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
import copy
import os
import shutil


def plot(bonds, mfe, mfe_dgv, cce, cce_dvg, e_nuc):

    fig, ax = plt.subplots(2)

    ax[0].plot(bonds, mfe- e_nuc, label = 'RHF')
    ax[0].plot(bonds, mfe_dgv- e_nuc, label = 'RHF (DG-V)')
    #ax[0].plot(bonds, cce, label = 'RHF')           
    #ax[0].plot(bonds, cce_dgv, label = 'RHF (DG-V)')
    ax[0].legend()
    ax[0].set_title('Meanfield Energy')

    ax[1].plot(bonds, mfe_dgv+ cce_dgv - e_nuc)
    ax[1].set_title('Electronic CCSD energy (DG-V)')

    plt.show()

def trans(x_, p_, a):
        '''
        counter clockwise a-rotation of p around x
        '''
        x, p = np.array(x_), np.array(p_)
        c, s  = np.cos(a), np.sin(a)
        R     = np.array(((c, -s), (s, c)))
        vec   = p - x
        r_vec = np.dot(R,vec)
        out = r_vec + x
        return out.tolist()

def mol_size(Mol):

    expan = lambda cords: max(cords) - min(cords)
    minim = lambda cords: min(cords)
    out   = []
    m     =[]
    for i in range(3):
        out.append(expan([mol[1][i] for mol in Mol]))
        m.append(minim([mol[1][i] for mol in Mol]))
    return np.array(out), m

if __name__ == '__main__':

        # clearing storage directory
        shutil.rmtree('output_h2')
        os.mkdir('output_h2')


        # Discretization of the box (defines the number of PW basis functions) 
        dgrid = [5]*3 
    
        # Bond distance of the H2, when computing pes this should be array
        bonds = np.linspace(1, 2.2, num= 3) 

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
        cell.basis = 'ccpvdz'                                                        
        cell.pseudo = 'gth-pade'                                                   
        cell.mesh = np.array([int(d * x) for d, x in zip(dgrid, X)])               
                                                                                   
        e_nuc = np.zeros(len(bonds))                                               
                                                                                   
        # Allocating solution vectors                                              
        mfe = np.zeros(len(bonds))                                                 
        mfe_dgv = np.zeros(len(bonds))                                             
                                                                                   
        cce = np.zeros(len(bonds))                                                 
        cce_dgv = np.zeros(len(bonds))                                             

        mad= np.zeros(len(bonds))
        tes= np.zeros(len(bonds))

        # Introducing lambda scaling factor for the ERI
        l_fact= 0.5 # NOTE here is the lambda factor
    
        for n, bd in enumerate(bonds):

            # Generating PySCF PBC object
            # atom position                                         
            Z = (X[0]-np.sum(bd))/2.0 + np.append( 0, np.cumsum(bd))
            cell.atom = [['H', (Z[0], X[1]/2, X[2]/2)],       
                         ['H', (Z[1], X[1]/2, X[2]/2)]]       

            cell.build()

            e_nuc[n]= cell.energy_nuc()

            # Checking condition number of active basis
            # NOTE DG-V basis is *always* well conditioned
            overlap = cell.pbc_intor('int1e_ovlp_sph')
            w, _ = la.eigh(overlap)
            cond_no = w[-1]/ w[0]

            if cond_no > 10000:
                # NOTE this happens rather fast when augmented basis functions
                # are added to the active basis, e.g. aug-cc-pVDZ or 6-31++G
                print('!!! WARNING !!!')
                print('Linear dependence detected in the active basis')
                print("Condition number: "    , cond_no)


            # VDG calculations
            # Voronoi DG Hamiltonian
            # NOTE The total number of DG-V basis functions per DG-V element/ cluster
            # can be directly controled by setting
            #
            #           dg_trunc = 'abs_num'
            #
            # and the number of kept basis functions is specified in
            #
            #           svd_tol = C
            #
            # So far, only n_k = C is possible. If needed I can adjust this,
            # s.t. n_k = C_k is possible.

            print("Creating  DG-V-" + cell.basis +  " Hamiltonian ...")
            cell_dgv= dg.dg_model_ham(cell, dg_cuts = None,
                                      dg_trunc = 'abs_num', svd_tol = 10, # NOTE scd_tol is the number of basis functions per cluster
                                      voronoi = True, dg_on=True,
                                      gram = None)

            cell_ref= dg.dg_model_ham(cell, dg_cuts = None,             
                                      dg_trunc = 'abs_num', svd_tol = 10,
                                      voronoi = True, dg_on=False,       
                                      gram = None)                      

            print('Done!')

            # Scaling the ERI  
            cell_dgv.eri= l_fact* cell_dgv.eri
            cell_ref.eri= l_fact* cell_ref.eri

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

            directory = 'output_h2/Param_' + str(n)
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
                        scipy.io.savemat(directory+ name_str +".mat",
                                         sub_eri_mdic)


            # RHF
            mfe_dgv[n] = cell_dgv.run_RHF()
            cce_dgv[n], _ = cell_dgv.run_CC(l_fact= l_fact)
            
            # Computing energy with 1-rdm
            #rdm1_dgv = cell_dgv.mf_dg.make_rdm1()
            #print(np.trace(rdm1_dgv@ h1e_vdg) +cell.energy_nuc() \
            #      +cell_dgv.tes)

            if False:            
                # computing the 1-rdm and energy directly from h1e_vdg
                # NOTE this only works for the l_fact= 0 case 
                rdm1_dgv = cell_dgv.mf_dg.make_rdm1()

                ham= h1e_vdg 
                ew, orbs= np.linalg.eigh(ham)        
                dm_ref= np.zeros_like(ham)
                for i in range(1):          
                    print('including orb: ', i)             
                    dm_ref += 2* np.outer(orbs[:,i], orbs[:,i]) 
                
                print('Difference ref to v-DG :', 
                        np.linalg.norm(rdm1_dgv- dm_ref))            
                print('Energy directly comp. from h1e:',
                        np.trace(dm_ref@ ham) +cell.energy_nuc() \
                        +cell_dgv.tes)
            
            # Reference computation
            mfe[n]= cell_ref.run_RHF()
            cce[n], _ = cell_ref.run_CC(l_fact= l_fact)      
 
            # Finite-size corrections
            mad[n]= cell_dgv.madelung
            tes[n]= cell_dgv.tes
 
        print(20*"#")
        print('Lambda factor : ', l_fact)
        print('AO basis      : ', cell.basis)
        print('Bond distances: ', bonds)
        print("Meanfield results:")
        print("Reference : ", mfe)
        print("DG-V      : ", mfe_dgv)
        print(5* "#")
        print('electronic meanfield result:')
        print("Reference : ", mfe- e_nuc)     
        print("DG-V      : ", mfe_dgv- e_nuc) 
        print("CCSD correlation energy:")
        print("Reference : ", cce)
        print("DG-V      : ", cce_dgv)
        print('Madlung constant')
        print(mad)
        print('Total energy shift due to Ewald probe charge')
        print('[-0.5* Nelec* madelung]')
        print(tes)
        print('Nuclear energy')
        print(e_nuc)

        plot(bonds, mfe, mfe_dgv, cce, cce_dgv, e_nuc)


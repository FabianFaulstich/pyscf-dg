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
        shutil.rmtree('output_h4')
        os.mkdir('output_h4')

        # angles in radian
        # NOTE that for the square geometry, a lot of DG-V basis functions
        # are needed due to the degenerate nature of the solution
        angles= np.linspace(1e-2, np.pi/ 2, num= 3)

        # Allocating solution vectors
        e_nuc = np.zeros(len(angles))

        mfe = np.zeros(len(angles))
        mfe_dgv = np.zeros(len(angles))

        # Introducing lambda scaling factor for the ERI
        l_fact= 0
    
        mad= np.zeros(len(angles))
        tes= np.zeros(len(angles)) 
    
        # Defining supercell size in which the molecule will be place
        # For simplicity we choose cubic supercell
        # NOTE We need sufficient vacuum package to avoid coupling
        # over the boundary
        sc_size = 12

        # Number of real-space gridpoints per unit length
        # For simplicity, we use a uniform grid
        dgrid   = [5] * 3

        # active basis
        basis   = '321g'

        # Initialize the H4 model (square geometry)
        # [Jankowski& Paldus, Int. J. of Quant. Chem., 18(5), 1243-1269 (1980)]
        # Bond-distance is 2 Bohr
        Mol_init = [['H', [0,0,0]],
                    ['H', [2,0,0]],
                    ['H', [0,2,0]],
                    ['H', [2,2,0]]]

        # discretization mesh
        mesh = [int(d * x) for d, x in zip(dgrid, [sc_size]*3)]

        for n, a in enumerate(angles):

            # Computing rotated H4
            Mol = copy.deepcopy(Mol_init)
            Mol[2][1][0:2] = trans(Mol_init[0][1][0:2],
                                   Mol_init[2][1][0:2], a)
            Mol[3][1][0:2] = trans(Mol_init[1][1][0:2],
                                   Mol_init[3][1][0:2], -a)

            # Centering Molecule in the supercell
            ms, mm = mol_size(Mol)

            offset = np.array([(sc_size- s)/2.- m for s, m in zip(ms, mm)])
            for k, off in enumerate(offset):
                for j in range(len(Mol)):
                    Mol[j][1][k] += off

            # Generating PySCF PBC object
            cell         = gto.Cell()
            cell.a       = [[sc_size, 0., 0.],
                            [0., sc_size, 0.],
                            [0., 0., sc_size]]
            cell.unit    = 'B'
            cell.verbose = 3
            cell.basis   = basis
            cell.pseudo  = 'gth-pade'
            cell.mesh    = np.array(mesh)
            cell.atom    = Mol
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
            # NOTE The total number of DG-V basis functions per DG-V element
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
                                      dg_trunc = 'abs_num', svd_tol = 8,
                                      voronoi = True, dg_on=True,
                                      gram = None)

            cell_ref= dg.dg_model_ham(cell, dg_cuts = None,             
                                      dg_trunc = 'abs_num', svd_tol = 2,
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

            directory = 'output_h4/Param_' + str(n)
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
            # Computing energy with 1-rdm
            #rdm1_dgv = cell_dgv.mf_dg.make_rdm1()
            #print(np.trace(rdm1_dgv@ h1e_vdg) +cell.energy_nuc() \
            #      +cell_dgv.tes)

            if True:            
                # computing the 1-rdm and energy directly from h1e_vdg
                # NOTE this only works for the l_fact= 0 case 
                rdm1_dgv = cell_dgv.mf_dg.make_rdm1()

                ham= h1e_vdg 
                ew, orbs= np.linalg.eigh(ham)        
                dm_ref= np.zeros_like(ham)
                for i in range(2):          
                    print('including orb: ', i)             
                    dm_ref += 2* np.outer(orbs[:,i], orbs[:,i]) 
                
                print('Difference ref to v-DG :', 
                        np.linalg.norm(rdm1_dgv- dm_ref))            
                print('Energy directly comp. from h1e:',
                        np.trace(dm_ref@ ham) +cell.energy_nuc() \
                        +cell_dgv.tes)
            
            mfe[n]= cell_ref.run_RHF()
       
            # Finite-size corrections
            mad[n]= cell_dgv.madelung
            tes[n]= cell_dgv.tes
 
        print(20*"#")
        print('Angles: ', angles)
        print("Meanfield results:")
        print("Reference : ", mfe)
        print("DG-V      : ", mfe_dgv)
        print('Madlung constant')
        print(mad)
        print('Total energy shift due to Ewald probe charge')
        print('[-0.5* Nelec* madelung]')
        print(tes)
        print('Nuclear energy')
        print(e_nuc)

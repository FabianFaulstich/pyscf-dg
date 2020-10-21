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
from pyscf import gto as molgto
from pyscf.pbc import dft as dft
from pyscf.pbc import gto, df, scf, mp, cc
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import FFTDF
from pyscf.pbc import tools
from pyscf import ao2mo
from sys import exit
import time

from scipy.optimize import curve_fit
import copy

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

    f   = open("out_tol-3_2.txt", "w")
    bs  = 5

    dgrid = [5,5,5] 
    atoms = 16

    bonds = np.array([1.8, 2.0, 2.4, 2.8, 3.2])
    tol = 1e-3

    n_ao        = np.zeros(len(bonds)) 
    n_ao_dg     = np.zeros(len(bonds))
    mfe         = np.zeros(len(bonds))
    mfe_dg      = np.zeros(len(bonds))
    #mpe         = np.zeros(len(bonds))
    #mpe_dg      = np.zeros(len(bonds))
    #cce         = np.zeros(len(bonds))
    #cce_dg      = np.zeros(len(bonds))



    basis = 'ccpvdz'
   
    for i, bond in enumerate(bonds):
        print('Computing H20 with equidinstant bonds: ', str(bond))

        Mol  = []
        Mol1 = []
        for n in range(atoms):
            Mol.append(['H', [n * bond, 0, 0]]) #1.4
            Mol1.append(['H', [n * 3.6, 0, 0]])

        # Centering Molecule in Box:
        ms, mm = mol_size(Mol1)
        ms[0] += 3.6
        boxsize = [ np.ceil(2*bs + s) for s in ms]
        ms, mm = mol_size(Mol)
        offset = np.array([ (bs - s)/2. - m for bs, s, m in zip(boxsize,ms,mm)])
        for k, off in enumerate(offset):
            for j in range(len(Mol)):
                Mol[j][1][k] += off
        del ms, mm 
    
        mesh = [int(d * x) for d, x in zip(dgrid, boxsize)]
    
        print("Mesh: ", mesh)
        print("BS:", boxsize)
        print("Basis", basis)
        print("Mol:", Mol)

        cell         = gto.Cell()
        cell.a       = [[boxsize[0], 0., 0.], 
                        [0., boxsize[1], 0.], 
                        [0., 0., boxsize[2]]]
        cell.unit    = 'B'
        cell.verbose = 3
        cell.basis   = basis
        cell.pseudo  = 'gth-pade'
        cell.mesh    = np.array(mesh)
        cell.atom    = Mol
        cell.build()
        
        print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
        start_dg = time.time()
        cell_vdg  = dg.dg_model_ham(cell, dg_cuts = None,
                        dg_trunc = 'abs_tol', svd_tol = tol, voronoi = True,
                        dg_on=True, gram = None)
        n_ao_dg[i] = cell_vdg.nao
        
        # HF in VDG
        mfe_dg[i] = cell_vdg.run_RHF()

        # MP2 in VDG
        #mpe_dg[i], _ = cell_vdg.run_MP2()

        # CCSD in VDG
        #cce_dg[i], _ = cell_vdg.run_CC()
        
        
        # HF in PySCF
        print('Creating Mean-field object:')

        mf = scf.RHF(cell, exxdiv='ewald') # madelung correction
        print('Running HF')

        mf.kernel()
        mfe[i] = mf.e_tot

        # MP2 in builtin
        #mpe[i], _ = mp.MP2(mf).kernel()

        # CCSD in builtin
        #cc_bi = cc.CCSD(mf)
        #cc_bi.kernel()
        #cce[i] =cc_bi.e_corr

        print()
        print('#########################')
        print('Hartree--Fock energy             :', mfe[i])
        print('Hartree--Fock energy (VdG)       :', mfe_dg[i])
        #print('----------')
        #print('MP2 corr. energy                 :', mpe[i])
        #print('MP2 corr. energy (VdG)           :', mpe_dg[i])
        #print('----------')
        #print('CCSD corr. energy                :', cce[i])
        #print('CCSD corr. energy (VdG)          :', cce_dg[i])
        print('----------')
        print('Number of VdG basis fct per atom :', n_ao_dg[i])
        print('SVD tollerance                   :', tol)
        print('#########################')
        print()

        del cell, cell_vdg, Mol, Mol1#, fftdf, mf, eri

    f.write("SV tolerance: " + str(tol) + "\n")
    f.write("Hydrogen chain H" + str(atoms) + "\n")
    f.write("Number of AO's (VdG): \n")
    f.write(str(n_ao_dg) + "\n")
    f.write('Hartree--Fock energy: \n')
    f.write(str(mfe) + '\n')
    f.write('Hartree--Fock energy (VdG): \n')
    f.write(str(mfe_dg) + '\n')
    #f.write('MP2 corr. energy: \n')
    #f.write(str(mpe) + '\n')
    #f.write('MP2 corr. energy (VdG): \n')
    #f.write(str(mpe_dg) + '\n')
    #f.write('CCSD corr. energy: \n')
    #f.write(str(cce) + '\n')
    #f.write('CCSD corr. energy (VdG): \n')
    #f.write(str(cce_dg))

    f.close()

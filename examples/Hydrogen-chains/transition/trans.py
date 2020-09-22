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

    f   = open("out.txt", "w")
    bs  = 5

    dgrid = [2,2,2] # 
    bond  = np.array([1.4])
    bond1 = np.array([3.6])
    #atoms = np.flip(np.linspace(2, 4, 2, dtype = int))
    atoms = np.flip(np.linspace(2,30,15, dtype = int))
    #atoms = np.linspace(2,20,10, dtype = int)
    atoms = np.array([28])
    
    svd_tol = np.array([1e-3, 1e-2, 1e-1])

    nnz_eri     = np.zeros(len(atoms))
    nnz_eri_dg  = np.zeros(len(atoms))
    n_lambda    = np.zeros(len(atoms))
    n_lambda_dg = np.zeros(len(atoms))
    n_ao        = np.zeros(len(atoms)) 
    n_ao_dg     = np.zeros(len(atoms))
    
    basis = 'ccpvdz'
   
    for tol in svd_tol:

        for i, no_atom in enumerate(atoms):
            Mol  = []
            Mol1 = []
            for n in range(no_atom):
                Mol.append(['H', [n * 1.4, 0, 0]])
                Mol1.append(['H', [n * 3.6, 0, 0]])

            # Centering Molecule in Box:
            ms, mm = mol_size(Mol1)
            ms[0] += 3.6
            boxsize = [ 2*bs + s for s in ms]
            ms, mm = mol_size(Mol)
            offset = np.array([ (bs - s)/2. - m for bs, s, m in zip(boxsize,ms,mm)])
            for k, off in enumerate(offset):
                for j in range(len(Mol)):
                    Mol[j][1][k] += off
          
            print("Box size: ", boxsize)
            print("Molecule: ", Mol)
            print("Offset: ", offset)
            print("Molecule length: ", ms)
            mesh = [int(d * x) for d, x in zip(dgrid, boxsize)]
           
            cell         = gto.Cell()
            cell.a       = [[boxsize[0], 0., 0.], [0., boxsize[1], 0.], [0., 0., boxsize[2]]]
            cell.unit    = 'B'
            cell.verbose = 3
            cell.basis   = basis
            cell.pseudo  = 'gth-pade'
            cell.mesh    = np.array(mesh)
            cell.atom    = Mol
            cell.build()

            dg_gramm, dg_idx = dg.get_dg_gramm(cell, None, 'abs_tol', tol, False, v_cells = None, v_net = None, dg_on=True)
            n_lambda_dg[i], nnz_eri_dg[i] = dg_tools.get_dg_nnz_eri(cell, dg_gramm, dg_idx) 

            n_ao[i]    = cell.nao
            n_ao_dg[i] = np.size(dg_gramm, 1) 

            mf = scf.RHF(cell, exxdiv = 'ewald')
            mf.kernel(dump_chk = False)

            eri = ao2mo.restore(1, mf._eri, cell.nao_nr()) 
            #h1e = mf.get_hcore()
            
            nnz_eri[i]  = np.count_nonzero(eri) # ~full eri
            n_lambda[i] = np.sum(np.abs(eri))   # ~full eri

            print("NNZ ERI: ", nnz_eri[i])
            print("NNZ ERI (DG): ", nnz_eri_dg[i])
            print("Lambda-value: ", n_lambda[i])
            print("Lambda-value (DG): ", n_lambda_dg[i])

        f.write("SV tolerance: " + str(tol) + "\n")
        f.write("Hydrogen chain to H" + str(atoms[0]) + "\n")
        f.write("Number of AO's: \n")
        f.write(str(np.flip(n_ao)) + "\n")
        f.write("nnz_eri: \n")
        f.write(str(np.flip(nnz_eri)) + "\n")
        f.write("Lambda: \n")
        f.write(str(np.flip(n_lambda)) + "\n")
        f.write("Number of AO's (DG): \n")
        f.write(str(np.flip(n_ao_dg)) + "\n")
        f.write("nnz_eri (DG): \n")
        f.write(str(np.flip(nnz_eri_dg)) + "\n")
        f.write("Lambda (DG): \n ")
        f.write(str(np.flip(n_lambda_dg)) + "\n")

    f.close()
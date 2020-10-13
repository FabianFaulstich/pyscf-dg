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

    f   = open("out_tol1e-2_n.txt", "w")
    bs  = 5

    dgrid = [2,2,2] # 
    bond  = np.array([1.4])
    bond1 = np.array([3.6])
    #atoms = np.flip(np.linspace(2, 4, 2, dtype = int))
    atoms = np.flip(np.linspace(2,30,15, dtype = int))
    #atoms = np.array([4])
    #print(atoms)
    #exit()
    #atoms = np.linspace(2,20,10, dtype = int)
    #atoms = np.array([26])
    
    svd_tol = np.array([1e-2])

    nnz_eri     = np.zeros(len(atoms))
    nnz_eri_pw  = np.zeros(len(atoms))
    nnz_eri_dg  = np.zeros(len(atoms))
    n_lambda    = np.zeros(len(atoms))
    n_lambda_dg = np.zeros(len(atoms))
    n_ao        = np.zeros(len(atoms)) 
    n_ao_dg     = np.zeros(len(atoms))
    
    basis = 'ccpvdz'
   
    for tol in svd_tol:
        for i, no_atom in enumerate(atoms):
            print("Computing H", no_atom)
            print("SVD tolerance: ", tol)

            Mol  = []
            Mol1 = []
            for n in range(no_atom):
                Mol.append(['H', [n * 1.4, 0, 0]])
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
            #print("Box size: ", boxsize)
            #print("Molecule: ", Mol)
            #print("Offset: ", offset)
            #print("Molecule length: ", ms)
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
            
            nnz_eri_pw[i] = np.prod(mesh)**2

            del boxsize
            
            print("Computing Gram Matrix ...")
            start = time.time()
            dg_gramm, dg_idx = dg.get_dg_gramm(
                    cell, None, 'abs_tol', tol, False, dg_on=True, gram = None)
            print("Done! Elapsed time: ", time.time() - start)
            
            print("Computing DG NNZ-ERI and Lambda value ...")
            start = time.time()
            n_lambda_dg[i], nnz_eri_dg[i] = dg_tools.get_dg_nnz_eri(
                    cell, dg_gramm, dg_idx) 
            print("Done! Elapsed time: ", time.time() -start)
            
            n_ao[i]    = cell.nao
            n_ao_dg[i] = np.size(dg_gramm, 1) 

            print("Computing HF ...")
            start = time.time()
            fftdf = df.FFTDF(cell)
            mf = scf.RHF(cell, exxdiv = 'ewald')
            mf.kernel(dump_chk = False)
            print("Done! Elapsed time: ", time.time() - start)
            
            print('Computing ERI tensor...')            
            start = time.time()
            eri = ao2mo.restore(1, fftdf.get_eri(), cell.nao_nr()) #mf._eri 
            print('Done! Elapsed time:', time.time() - start)
            #h1e = mf.get_hcore()
            eri[np.abs(eri) < 1e-6] = 0
            
            nnz_eri[i]  = np.count_nonzero(eri) 
            n_lambda[i] = np.sum(np.abs(eri))   

            print("NNZ ERI: ", nnz_eri[i])
            print("NNZ ERI (DG): ", nnz_eri_dg[i])
            print("Lambda-value: ", n_lambda[i])
            print("Lambda-value (DG): ", n_lambda_dg[i])


            del cell, Mol, Mol1, fftdf, mf, eri

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
        f.write("nnz_eri (PW): \n")
        f.write(str(np.flip(nnz_eri_pw)) + "\n")

    f.close()

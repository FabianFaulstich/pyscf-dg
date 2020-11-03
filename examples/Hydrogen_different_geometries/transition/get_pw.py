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
from pyscf.pbc.df import fft
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

    f   = open("out_pw.txt", "w")
    
    bs    = 5
    dgrid = [5,5,5] # 
    atoms = np.linspace(2,30,15, dtype = int)
    
    nnz_eri_pw  = np.zeros(len(atoms))
    n_lambda    = np.zeros(len(atoms))
    
    basis = 'ccpvdz'
   
    for i, no_atom in enumerate(atoms):
        print("Computing H", no_atom)

        Mol  = []
        Mol1 = []
        for n in range(no_atom):
            Mol.append(['H', [n * 1.7, 0, 0]]) #1.4
            Mol1.append(['H', [n * 3.6, 0, 0]])

        # Centering Molecule in Box:
        ms, mm = mol_size(Mol1)
        ms[0] += 3.6
        boxsize = [ np.ceil(2*bs + s) for s in ms]
        ms, mm = mol_size(Mol)
        offset = np.array([(bs - s)/2.- m for bs, s, m in zip(boxsize,ms,mm)])
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
        #cell.ke_cutoff = 20
        cell.basis   = basis
        cell.pseudo  = 'gth-pade'
        cell.mesh    = np.array(mesh)
        cell.atom    = Mol
        cell.build()
       
        mesh = cell.mesh
        vol = cell.vol
        ngrids = np.prod(mesh)
        dvol = vol / ngrids
        
        coulG = tools.get_coulG(cell)
        dd = np.zeros(len(coulG))
        for i in range(len(coulG)):
            d = np.zeros(len(coulG))
            d[i] = 1
            tools.fft(d, mesh) * dvol
            vcoul = coulG *  d
            dd = np.abs(tools.ifft(vcoul, mesh).real / dvol)

        n_lambda[i]   = np.sum(np.abs(coulG))
        print(np.sum(np.abs(dd)))
        nnz_eri_pw[i] = np.prod(mesh)**2
         
        print("NNZ ERI (PW): ", nnz_eri_pw[i])
        print("Lambda  (PW): ", n_lambda[i])

        del cell, Mol, Mol1#, fftdf, mf, eri
    
    print("NNZ ERI (PW): ", nnz_eri_pw)
    print("Lambda  (PW): ", n_lambda)

    f.write("Hydrogen chain to H" + str(atoms[0]) + "\n")
    f.write("Lambda (PW): \n")
    f.write(str(np.flip(n_lambda)) + "\n")
    f.write("nnz_eri (PW): \n")
    f.write(str(np.flip(nnz_eri_pw)) + "\n")

    f.close()

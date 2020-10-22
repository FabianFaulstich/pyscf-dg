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

    f   = open("out_la_pw.txt", "w")
    bs  = 5

    dgrid = [2,2,2] # 
    #bond  = np.array([1.7])
    #bond1 = np.array([3.6])
    #atoms = np.flip(np.linspace(2, 4, 2, dtype = int))
    #atoms = np.flip(np.linspace(2,30,15, dtype = int))
    atoms = np.linspace(2,30,15, dtype = int)
    #atoms = np.array([8])
    #atoms = np.linspace(2, 8, num = 4, dtype = int)
    #print(atoms)
    #exit()
    #atoms = np.linspace(2,20,10, dtype = int)
    #atoms = np.array([26])
    
    svd_tol = np.array([1e-2]) # chenge back to 1e-1

    nnz_eri     = np.zeros(len(atoms))
    nnz_eri_pw  = np.zeros(len(atoms))
    nnz_eri_dg  = np.zeros(len(atoms))
    n_lambda    = np.zeros(len(atoms))
    n_lambda_dg = np.zeros(len(atoms))
    n_ao        = np.zeros(len(atoms)) 
    n_ao_dg     = np.zeros(len(atoms))
    
    basis = '631++g**'
   
    for tol in svd_tol:
        for i, no_atom in enumerate(atoms):
            print("Computing H", no_atom)
            print("SVD tolerance: ", tol)

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

            nnz_eri_pw[i] = np.prod(mesh)**2
            
            # Creating mol object to fetch ao_projections

            mol       = molgto.Mole()
            mol.atom  = Mol
            mol.basis = basis
            mol.unit  = 'B'
            mol.build()

            coords = cell.get_uniform_grids()
            ao_values = mol.eval_gto("GTOval_sph", coords)
            U, _, VT = la.svd(ao_values, full_matrices = False)
            
            # U0 describes L^2 normalized functions.
            # This is only necesary for using dg_tools.get_dg_nnz_eri(), because 
            # dg_model_ham.gram is stored in that way. This will be changed in 
            # future adaptation of the code. Direct input of the projections 
            # matrix should always only consist of the nodel values of the 
            # primitive basis. Note that the gram input for a dg_model_ham object             # takes the nodel values of the primitive basis as input for gram. 
                       
            U0 = np.dot(U, VT) / np.sqrt(dvol)
            del boxsize
           
            print('computing la for H', no_atom)
            la[i] = dg_tools.get_la_pw(cell, cell.get_uniform_grids())
            print('computed la: ', la[i])

            del cell, Mol, Mol1#, fftdf, mf, eri

        f.write("Hydrogen chain to H" + str(atoms[0]) + "\n")
        f.write("lambda value (PW): \n")
        f.write(str(la))

    f.close()

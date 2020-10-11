import sys
sys.path.append('../../src')
import time
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.scf as pbcscf
from pyscf.pbc import  mp, cc

import dg_model_ham as dg
import dg_tools
import numpy as np

def read_sc():
    
    f    = open('input_file_small.txt', 'r')
    sc   = [] 
    sc_b = False
    for line in f:
        if sc_b:
            fl = line.split()
            el = [float(x) for x in fl]
            sc.append(el)
        
        if 'Supercell' in line:
            sc_b = True
    return sc

def read_mol():

    f   = open('input_file_small.txt', 'r')
    mol = [] 
    for line in f:
        if 'atom:' in line:
            atom = []
            fl = line.split()[1]
            atom.append(fl)
        if 'pos:' in line:
            fl  = line.split()[1:]
            pos = [float(x) for x in fl]
            atom.append(pos)
            mol.append(atom)

    return mol

if __name__ == '__main__':


    mol    = read_mol()
    cell_a = read_sc()
    
    boxsize   = []
    for i, elem in enumerate(cell_a):
        boxsize.append(elem[i])
    print(boxsize)
    dgrid      = [25, 25, 15]

    # discretization mesh
    mesh = [int(d * x) for d, x in zip(dgrid, boxsize)]
    
    #print(cell_a)
    #print(mesh)
    #exit()

    cell = pbcgto.Cell()
    cell.verbose = 3
    cell.atom = mol
    cell.a = cell_a
    #cell.dimension=2
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = np.array(mesh)
    cell.build()


    # Computing pyscf HF
    print('Creating Mean-field object:')
    
    mf = pbcscf.RHF(cell, exxdiv='ewald') # madelung correction
    print('Running HF')
    
    mf.kernel()
    mfe = mf.e_tot/2 #seems to be some weird dependence on particle number ?

    print('AO Basis          : ', cell.basis)
    print('Supercell         : ', boxsize[0],' x ',boxsize[1],' x ',boxsize[2])
    print('Discretizatio     : ', mesh[0]   ,' x ',mesh[1]   ,' x ',mesh[2])
    print('Mean-field energy : ', mfe)

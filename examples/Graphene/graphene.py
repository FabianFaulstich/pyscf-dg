import sys
sys.path.append('../../src')
import time
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.scf as pbcscf

import dg_model_ham as dg
import dg_tools
import numpy as np

def read_sc():
    
    f    = open('input_file.txt', 'r')
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

    f   = open('input_file.txt', 'r')
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
    
    #log = open("log.txt", "w")
    #f   = open("out.txt", "w")

    cell = pbcgto.Cell()
    cell.verbose = 3
    cell.atom = mol
    cell.a = cell_a
    cell.basis = 'sto3g'
    cell.pseudo = 'gth-pade'
    cell.build()

    #exit()
    # DG calculations

    # Customizing gram matrix for dg calculations
    # Fetching AO's
    #gram_mat = pbcdft.numint.eval_ao(cell, cell.get_uniform_grids(), deriv=0)

    # Computiong MO coeff's
    print('Creating Mean-field object:')
    mf = pbcscf.RHF(cell, exxdiv='ewald') # madelung correction
    print('Running HF')
    mf.kernel()
    mos = mf.mo_coeff 
    #gram_mo = np.dot(gram_mat, mos[:,:4])

    #print(gram_mo.shape)
    #print(gram_mat.shape)
    print(mos.shape)

    print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
    start_dg = time.time()
    #cell_dg  = dg.dg_model_ham(cell, dg_cuts = None, dg_trunc = 'abs_tol',
    #        svd_tol = 1e-3, voronoi = True, dg_on=True, gram = gram_mo)





import sys
sys.path.append('../../src')
import time
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.tools import pyscf_ase
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.scf as pbcscf

from pyscf.pbc.tools import pyscf_ase
from ase.lattice.hexagonal import *
from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape
from ase.build import make_supercell
from ase.visualize import view
from ase.io import read, write

import dg_model_ham as dg
import dg_tools
import numpy as np

def get_graphene_input():

    print('Generate one primitive cell.')
    index1=index2=1
    lat_a = 2.45
    lat_c = 10.0

    gra = Graphene(symbol = 'C',latticeconstant={'a':lat_a,'c':lat_c},
            size=(index1,index2,1))

    print('Use find_optimal_cell_shape to turn it into an orthorhombic cell.')
    '''
    https://wiki.fysik.dtu.dk/ase/tutorials/defects/defects.html#algorithm-for-finding-optimal-supercell-shapes
    
    For graphene, the smallest copy contains 2 primitive cells (4 atoms).
    Should get
    P1 = array([[1, 0, 0],
            [1, 2, 0],
            [0, 0, 1]])
    '''
    #P1 = find_optimal_cell_shape(gra.cell, 2, 'sc')
    #print(P1)
    P1 = np.array([[1, 0, 0],
           [1, 2, 0],
           [0, 0, 1]])

    #print('make a supercell')
    # the supercell can also be constructed as
    # P1 @ gra.cell
    # the cell vectors are given as the row vectors
    supercell = make_supercell(gra, P1)

    #print('Supercell.cell = ', supercell.cell)
    #print('P1 @ gra.cell  = ', P1 @ gra.cell.array)
    #print(supercell.arrays)

    # generate a large supercell
    supercell = supercell * [2,2,1]
    view(supercell)

    #print('Convert to pyscf')
    import pyscf.pbc.gto as pbcgto
    import pyscf.pbc.dft as pbcdft
    import pyscf.pbc.scf as pbcscf
    from pyscf.pbc.tools import pyscf_ase

    mol   = pyscf_ase.ase_atoms_to_pyscf(supercell)
    cell_a = supercell.cell

    write_mol_to_file(mol)
    write_sc_to_file(cell_a)

    return mol, cell_a

def write_sc_to_file(sc):

    f = open('input_file.txt', 'a')
    f.write('Supercell: \n')
    for elem in sc:
        for el in elem:
            f.write(str(el) + ' ')
        f.write('\n')
    f.close()

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

def write_mol_to_file(mol):

    f = open('input_file.txt', 'w')
    for atom in mol:
        f.write('atom: ' + atom[0] + '\n')
        f.write('pos: ')
        for elem in atom[1]:
            f.write(str(elem) + ' ')
        f.write('\n')
    f.close()

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
    #mol, cell_a = get_graphene_input()

    #print(mol)
    #print(cell_a)
    
    #exit()
    #log = open("log.txt", "w")
    #f   = open("out.txt", "w")

    cell = pbcgto.Cell()
    cell.verbose = 5
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





from pyscf.pbc.tools import pyscf_ase
from ase.lattice.hexagonal import *
from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape
from ase.build import make_supercell
from ase.visualize import view
from ase.io import read, write
import numpy as np

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
    #supercell = supercell * [2,2,1]
    view(supercell)

    mol   = pyscf_ase.ase_atoms_to_pyscf(supercell)
    cell_a = supercell.cell

    write_mol_to_file(mol)
    write_sc_to_file(cell_a)


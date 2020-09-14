'''
Demonstrate the (complicated way) of generating a graphene example using
ASE, and turn it into an orthorhombic cell, and then convert to the
pyscf format.

Everything is in the unit of angstrom.
'''

import numpy as np
from ase.lattice.hexagonal import *
from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape  
from ase.build import make_supercell
from ase.visualize import view
from ase.io import read, write

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
P1 = np.array([[1, 0, 0],
       [1, 2, 0],
       [0, 0, 1]])


print('make a supercell')
# the supercell can also be constructed as
# gra.cell * P1.T
# the cell vectors are given as the row vectors
supercell = make_supercell(gra, P1)

print('Supercell.cell = ', supercell.cell)
print('gra.cell*P1.T = ', gra.cell * P1.T)
print(supercell.arrays)

# generate a large supercell
supercell_big = supercell * [4,4,1]
view(supercell_big)



print('Convert to pyscf')
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.scf as pbcscf
from pyscf.pbc.tools import pyscf_ase

cell = pbcgto.Cell()
cell.verbose = 5
cell.atom=pyscf_ase.ase_atoms_to_pyscf(supercell)
cell.a=supercell.cell
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.build()

#mf=pbcdft.RKS(cell)
#
#mf.xc='lda,vwn'

# Note: unit = angstrom
print('cell.unit = ', cell.unit)
#mf=pbcscf.RHF(cell)
#
#print(mf.kernel())

print('Dump out the supercell')
write('graphene.xyz', supercell)

import numpy as np
import time

import pyscf
from pyscf import lib
from pyscf.pbc import dft as dft
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import FFTDF
from pyscf.pbc import tools
from pyscf import ao2mo


# an example of H4 molecul
Mol = [['H',[1, 1, 0]], ['H',[2, -1, 0]], ['H',[-.5, -1.5, 0]], ['H',[-1, .25, 0]]]

atom_pos = np.array([atom[1] for atom in Mol])
atoms    = [atom[0] for atom in Mol]
Mol_size = np.array([np.amax(atom_pos[:,i])-np.amin(atom_pos[:,i]) for i in range(3)])
offset   = np.array([6,6,6])
atom_off = np.array([offset[d]/2.0 - np.amin(atom_pos[:,d]) for d in range(3)])

X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
atom_box = np.array([pos + atom_off for pos in atom_pos])
Mol_box  = [[atoms[i],atom_box[i]] for i in range(len(atoms))]

dgrid = [2]*3

cell = gto.Cell()
cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
cell.unit    = 'B'
cell.verbose = 3
cell.basis   = 'sto-3g' #gth-dzvp
cell.pseudo  = 'gth-pade'
# potential speed up 
cell.ke_cutoff = 200.0
cell.mesh    = np.array([int(d * x) for d, x in zip(dgrid, X)])
cell.atom    = Mol_box
cell.build()

# HF
print("Computing HF in " + cell.basis +  " basis ...")
start_hf = time.time()
mf = scf.RHF(cell, exxdiv='None') # madelung correction: ewlad
mf.kernel()
mfe = mf.e_tot
end_hf   = time.time()
print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
print()

import numpy as np
from pyscf import gto, scf, cc

Mol = [['H',[1, 1, 0]], ['H',[2, -1, 0]], ['H',[-.5, -1.5, 0]], ['H',[-1, .25, 0]]]

atom_pos = np.array([atom[1] for atom in Mol])
atoms    = [atom[0] for atom in Mol]
Mol_size = np.array([np.amax(atom_pos[:,i])-np.amin(atom_pos[:,i]) for i in range(3)])
offset   = np.array([3,3,3])
atom_off = np.array([offset[d]/2.0 - np.amin(atom_pos[:,d]) for d in range(3)])

X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
atom_box = np.array([pos + atom_off for pos in atom_pos])
Mol_box  = [[atoms[i],atom_box[i]] for i in range(len(atoms))]

mol = gto.M()
mol.basis = 'sto-3g'
mol.atom = Mol_box
mol.unit = 'bohr'
mol.build()

mf        = scf.RHF(mol).run()
mfe    = mf.e_tot
mycc      = cc.CCSD(mf)
mycc.kernel()
ecc    = mycc.e_tot

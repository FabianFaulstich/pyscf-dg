from pyscf import gto, scf, cc
import numpy as np
import matplotlib                                                               
import matplotlib.pyplot as plt   

Mol = [['H',[-2.4, -1.86, 0]], ['H',[-2.4, 1.86, 0]], ['C',[-1.34, 0, 0]],  
           ['C',[1.34, 0, 0]], ['H',[2.4,-1.86, 0]], ['H',[2.4,1.86,0 ]]]
# dissociation in x-direction relative to the optimal geometry              
rel_dis = np.array([ -.6, -.4, -.3, -.2, -.1,  0, .1, .2, .3, .4, .6, 1.0, 1.4]) 
mfe     = np.zeros(len(rel_dis))
ecc     = np.zeros(len(rel_dis))
for i, diss in enumerate(rel_dis):
    # Placing Molecule in the box:
    atom_pos = np.array([atom[1] for atom in Mol])
    atom_pos[int(len(atom_pos)/2):,0] += diss
    atoms    = [atom[0] for atom in Mol]
    offset    = np.array([6,6,6])
    atom_off  = np.array([offset[d]/2.0 - np.amin(atom_pos[:,d]) for d in range(3)])
    atoms_box = np.array([pos + atom_off for pos in atom_pos])
    atoms_box = np.array([pos + atom_off for pos in atom_pos])
    Mol_box  = [[atoms[i],atoms_box[i]] for i in range(len(atoms))]
    
    mol = gto.M()
    mol.basis = 'cc-pvdz'
    mol.atom = Mol_box
    mol.unit = 'bohr'
    mol.build()

    mf        = scf.RHF(mol).run()
    mfe[i]    = mf.e_tot
    mycc      = cc.CCSD(mf)
    mycc.kernel()
    ecc[i]    = mycc.e_tot 



plt.plot(rel_dis + 2.68, mfe, 'b-v', label =  'HF  (' + mol.basis + ')')
plt.plot(rel_dis + 2.68, ecc, 'b-x', label =  'CC  (' + mol.basis + ')')
plt.legend()
plt.show()

exit()
mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)
mycc.kernel()

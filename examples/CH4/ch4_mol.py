from pyscf import gto, scf, mp, cc, ao2mo, fci
from pyscf.fci import direct_spin0

import time
import copy
import numpy as np

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

    log = open("log_mol.txt", "w")
    f   = open("out_mol.txt", "w")
    f.write("Computing CH4 in optimal Geometry:\n")
    log.write("Computing CH4 in optimal Geometry:\n")

    Mol = [['C', [ 0, 0, 0]],
           ['H', [ 1.186, 1.186, 1.186]], 
           ['H', [ 1.186,-1.186,-1.186]], 
           ['H', [-1.186, 1.186,-1.186]], 
           ['H', [-1.186,-1.186, 1.186]]]

    mol = gto.M()
    mol.basis = 'sto3g'
    mol.atom  = Mol
    mol.unit  = 'bohr'
    mol.verbose = 3
    mol.build()

    # HF
    log.write("        Comuting HF using PyScf...\n")
    start = time.time()

    mf = mol.RHF()
    mf.kernel()
    mfe = mf.e_tot

    log.write("        Done! Elapsed time: " +
              str(time.time() - start) + "sec.\n")

    # MP2
    log.write("        Comuting MP2 using PyScf...\n")
    start = time.time()

    emp, _ = mp.MP2(mf).kernel()

    log.write("        Done! Elapsed time: " +
              str(time.time() - start) + "sec.\n")


    # CCSD
    log.write("        Comuting CCSD using PyScf...\n")
    start = time.time()

    mycc = cc.CCSD(mf)
    mycc.kernel()
    ecc = mycc.energy()

    log.write("        Done! Elapsed time: " +
              str(time.time() - start) + "sec.\n")

    f.write("AO Basis              : " + mol.basis   + "\n")
    f.write("Mean-field energy     : " + str(mfe)    + "\n")
    f.write("MP2 corr. energy      : " + str(emp)    + "\n")
    f.write("CCSD corr. energy      : " + str(ecc)    + "\n")
    
    print('##################################')
    print('AO Basis              : ', mol.basis)
    print('Mean-field energy     : ', mfe)
    print('MP2 corr. energy      : ', emp)
    print('CCSD corr. energy     : ', ecc)

    f.close()
    log.close()

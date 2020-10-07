import sys
sys.path.append('../../src')

import dg_model_ham as dg
import dg_tools

import pyscf
from pyscf import lib
from pyscf.pbc import dft as dft
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import FFTDF
from pyscf.pbc import tools
from pyscf import ao2mo
from sys import exit

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

    log = open("log.txt", "w")
    f   = open("out.txt", "w")
    f.write("Computing CH4 in optimal Geometry:\n")
    log.write("Computing CH4 in optimal Geometry:\n")

    boxsizes   = [18] * 3
    dgrid      = [10] * 3
    bases      = 'sto3g'
    accuracies = .99

    Mol_init = [['C', [ 0, 0, 0]],
                ['H', [ 1.186, 1.186, 1.186]], 
                ['H', [ 1.186,-1.186,-1.186]], 
                ['H', [-1.186, 1.186,-1.186]], 
                ['H', [-1.186,-1.186, 1.186]]]

    Mol = copy.deepcopy(Mol_init)
   
    # discretization mesh
    mesh = [int(d * x) for d, x in zip(dgrid, boxsizes)]

    # Centering Molecule in Box:
    ms, mm = mol_size(Mol)
    offset = np.array([ (bs - s)/2. - m for bs, s, m in zip(boxsizes, ms, mm)])
    
    for k, off in enumerate(offset):
        for j in range(len(Mol)):
            Mol[j][1][k] += off
    
    cell         = gto.Cell()
    cell.a       = [[boxsizes[0], 0., 0.], 
                    [0., boxsizes[1], 0.],
                    [0., 0., boxsizes[2]]]
    cell.unit    = 'B'
    cell.verbose = 3
    cell.basis   = bases
    cell.pseudo  = 'gth-pade'
    cell.mesh    = np.array(mesh)
    cell.atom    = Mol
    cell.build()

    # HF in builtin
    log.write("        Comuting HF using PyScf-PBC...\n")
    start = time.time()

    mf = scf.RHF(cell, exxdiv='ewald') # madelung correction
    mf.kernel(dump_chk = False)
    mfe = mf.e_tot

    log.write("        Done! Elapsed time: " +
              str(time.time() - start) + "sec.\n")

    exit()

    cell_vdg = dg.dg_model_ham(cell, None ,'rel_num', accuracies, True)
    # HF in VDG
    log.write("        Computing HF in VDG Bases ...\n")
    start = time.time()

    mfe_dg = cell_vdg.run_RHF()

    log.write("        Done! Elapsed time: " +
    str(time.time() - start) + "sec.\n")

    # MP2 in VDG
    log.write("        Computing MP2 in VDG Bases ...\n")
    start = time.time()

    mpe_dg, _ = cell_vdg.run_MP2()

    log.write("        Done! Elapsed time: " +
              str(time.time() - start) + "sec.\n")

    del cell_vdg

    # HF in builtin
    log.write("        Comuting HF using PyScf-PBC...\n")
    start = time.time()

    mf = scf.RHF(cell, exxdiv='ewald') # madelung correction
    mf.kernel(dump_chk = False)
    mfe = mf.e_tot

    log.write("        Done! Elapsed time: " +
              str(time.time() - start) + "sec.\n")

    # MP2 in builtin
    log.write("        Comuting MP2 using PyScf-PBC...\n")
    start = time.time()

    mpe, _ = mp.MP2(mf).kernel()

    log.write("        Done! Elapsed time: " +
              str(time.time() - start) + "sec.\n")
    
    del cell

    f.write("AO Basis              : " + bases   + "\n")
    f.write("Mean-field energy     : " + str(mfe)    + "\n")
    f.write("MP2 corr. energy      : " + str(mpe)    + "\n")

    f.write("Mean-field energy (DG): " + str(mfe_dg)    + "\n")
    f.write("MP2 corr. energy  (DG): " + str(mpe_dg)    + "\n")

    print('##################################')
    print('AO Basis              : ', bases)
    print('Mean-field energy     : ', mfe)
    print('MP2 corr. energy      : ', mpe)
    print('Mean-field energy (DG): ', mfe_dg)
    print('MP2 corr. energy  (DG): ', mpe_dg)

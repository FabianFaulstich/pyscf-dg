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
    
    log = open("log.txt", "w")
    f   = open("out.txt", "w")
    
    boxsizes   = []
    for i, elem in enumerate(cell_a):
        boxsizes.append(elem[i])
    print(boxsizes)
    dgrid      = [25, 25, 15]

    # discretization mesh
    mesh = [int(d * x) for d, x in zip(dgrid, boxsizes)]
    #print(cell_a)
    #print(mesh)
    #exit()

    cell = pbcgto.Cell()
    cell.verbose = 3
    cell.atom = mol
    cell.a = cell_a
    #cell.dimension=2
    cell.basis = 'ccpvdz'
    cell.pseudo = 'gth-pade'
    cell.mesh = np.array(mesh)
    cell.build()

    # DG calculations

    # Customizing gram matrix for dg calculations
    # Fetching AO's
    gram_mat = pbcdft.numint.eval_ao(cell, cell.get_uniform_grids(), deriv=0)

    # Computiong MO coeff's
    print('Creating Mean-field object:')
    
    mf = pbcscf.RHF(cell, exxdiv='ewald') # madelung correction
    print('Running HF')
    
    mf.kernel()
    mfe = mf.e_tot
    #exit()

    nelec = cell.tot_electrons()
    bands = np.linspace(2, 40, num = 20)
    #bands = np.array([2, 4])
    mos = mf.mo_coeff 

    # MP2 in builtin
    log.write("        Comuting MP2 using PyScf-PBC...\n")
    start = time.time()

    mpe, _ = mp.MP2(mf).kernel()

    log.write("        Done! Elapsed time: " +
              str(time.time() - start) + "sec.\n")

    mfe_dg = np.zeros(len(bands))
    mpe_dg = np.zeros(len(bands))
    
    for i, bds in enumerate(bands):
        print("Kept MO's: ", int(nelec + bds))
        gram_mo = np.dot(gram_mat, mos[:,: int(nelec + bds)])

        print(gram_mo.shape)
        print(gram_mat.shape)
        print(mos.shape)

        print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
        start_dg = time.time()
        cell_vdg  = dg.dg_model_ham(cell, dg_cuts = None, dg_trunc = 'abs_tol',
                svd_tol = 1e-3, voronoi = True, dg_on=True, gram = gram_mo)

        # HF in VDG
        log.write("        Computing HF in VDG Bases ...\n")
        start = time.time()

        mfe_dg[i] = cell_vdg.run_RHF()

        log.write("        Done! Elapsed time: " +
        str(time.time() - start) + "sec.\n")

        # MP2 in VDG
        log.write("        Computing MP2 in VDG Bases ...\n")
        start = time.time()

        mpe_dg[i], _ = cell_vdg.run_MP2()

        log.write("        Done! Elapsed time: " +
                  str(time.time() - start) + "sec.\n")
        print('AO Basis              : ', cell.basis)
        print('Mean-field energy     : ', mfe)
        print('MP2 corr. energy      : ', mpe)
        print('Kepts MOs             : ', bands)
        print('Mean-field energy (DG): ', mfe_dg)
        print('MP2 corr. energy  (DG): ', mpe_dg)
        del cell_vdg

    f.write("AO Basis              : " + cell.basis   + "\n")
    f.write("Mean-field energy     : " + str(mfe)    + "\n")
    f.write("MP2 corr. energy      : " + str(mpe)    + "\n")

    f.write("Bands included in DG  : " + str(bands) + "\n")
    f.write("Mean-field energy (DG): " + str(mfe_dg)    + "\n")
    f.write("MP2 corr. energy  (DG): " + str(mpe_dg)    + "\n")


    print('##################################')
    print('AO Basis              : ', cell.basis)
    print('Mean-field energy     : ', mfe)
    print('MP2 corr. energy      : ', mpe)
    print('Kepts MOs             : ', bands)
    print('Mean-field energy (DG): ', mfe_dg)
    print('MP2 corr. energy  (DG): ', mpe_dg)

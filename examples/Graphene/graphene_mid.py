import sys
sys.path.append('../../src')
import time
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.scf as pbcscf
from pyscf.pbc import  mp, cc
from pyscf.pbc import tools

import dg_model_ham as dg
import dg_tools
import numpy as np

def read_sc():
    
    #f    = open('input_file_small.txt', 'r')
    f    = open('input_file_mid.txt', 'r')
    #f    = open('input_file_large.txt', 'r')
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

    #f   = open('input_file_small.txt', 'r')
    f   = open('input_file_mid.txt', 'r')
    #f   = open('input_file_large.txt', 'r')
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
    f   = open("out_mid.txt", "w")
    
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

    # mesh from pyscf
    mesh = [30, 30, 40]

    cell = pbcgto.Cell()
    cell.unit = 'a'
    cell.a = cell_a
    cell.atom = mol
    cell.mesh = np.array(mesh)
    cell.dimension=2
    cell.pseudo = 'gth-pade'
    cell.verbose = 3
    cell.basis = 'gth-dzvp' #'gth-szv' (try with dzvp, tzvp, tzv2p)
    cell.precision = 1e-6
    cell.build()

    cell_mos  = pbcgto.copy(cell)

    # DG calculations
    nelec = cell.tot_electrons()
    natom = len(mol)
    # Customizing gram matrix for dg calculations
    # Fetching AO's
    gram_mat = pbcdft.numint.eval_ao(cell_mos, cell_mos.get_uniform_grids(),
            deriv=0)

    # Computiong MO coeff's
    print('Creating Mean-field object:')
    
    mf = pbcscf.RHF(cell_mos, exxdiv='ewald') # madelung correction
    print('Running HF')
    
    mf.kernel()
    mfe = mf.e_tot/natom
    left_occs = nelec - nelec/natom  
    occs = np.linspace(0, left_occs, num = left_occs+1)
    #bands = np.array([2, 4])
    mos = mf.mo_coeff

    # MP2 in builtin
    log.write("        Comuting MP2 using PyScf-PBC...\n")
    start = time.time()

    #mpe, _ = mp.MP2(mf).kernel()
    #mpe = mpe/ natom

    log.write("        Done! Elapsed time: " +
              str(time.time() - start) + "sec.\n")

    mfe_dg = np.zeros(len(occs))
    mpe_dg = np.zeros(len(occs))
    bf_dg  = np.zeros(len(occs))

    tol = 1e-1
    
    for i, noccs in enumerate(occs):
        nmos = int(nelec/natom + noccs)
        print("Kept MO's: ", nmos) 
        gram_mo = np.dot(gram_mat, mos[:,:nmos])

        print(gram_mo.shape)
        print(gram_mat.shape)
        print(mos.shape)

        print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
        start_dg = time.time()
        cell_in  = pbcgto.copy(cell) 
        cell_vdg  = dg.dg_model_ham(cell_in, dg_cuts = None, 
                dg_trunc = 'abs_tol', svd_tol = tol, voronoi = True, 
                dg_on=True, gram = gram_mo)
        bf_dg[i] = cell_vdg.nao/natom
        # HF in VDG
        log.write("        Computing HF in VDG Bases ...\n")
        start = time.time()

        mfe_dg[i] = cell_vdg.run_RHF()/ natom

        log.write("        Done! Elapsed time: " +
        str(time.time() - start) + "sec.\n")

        # MP2 in VDG
        log.write("        Computing MP2 in VDG Bases ...\n")
        start = time.time()

        #mpe_dg[i], _ = cell_vdg.run_MP2()
        #mpe_dg[i] = mpe_dg[i]/natom
        log.write("        Done! Elapsed time: " +
                  str(time.time() - start) + "sec.\n")
        print('AO Basis              : ', cell.basis)
        print('Mean-field energy     : ', mfe)
        #print('MP2 corr. energy      : ', mpe)
        print('Kepts MOs             : ', nmos)
        print('No VdG bf per elem    : ', bf_dg[i])
        print('Mean-field energy (DG): ', mfe_dg[i])
        #print('MP2 corr. energy  (DG): ', mpe_dg[i])
        del cell_vdg

    f.write("AO Basis              : " + cell.basis   + "\n")
    f.write("Mean-field energy     : " + str(mfe)    + "\n")
    #f.write("MP2 corr. energy      : " + str(mpe)    + "\n")

    f.write("MO's included in DG   : " + str(occs) + "\n")
    f.write("SVD tollerance        : " + str(tol) + "\n")
    f.write("Number of VdG bfs     : " + str(bf_dg) + "\n")
    f.write("Mean-field energy (DG): " + str(mfe_dg)    + "\n")
    #f.write("MP2 corr. energy  (DG): " + str(mpe_dg)    + "\n")


    print('##################################')
    print('AO Basis              : ', cell.basis)
    print('Mean-field energy     : ', mfe)
    #print('MP2 corr. energy      : ', mpe)
    print('Kepts MOs             : ', occs)
    print('No VdG bf per elem    : ', bf_dg)
    print('Mean-field energy (DG): ', mfe_dg)
    #print('MP2 corr. energy  (DG): ', mpe_dg)

import sys
sys.path.append('../../src')

import matplotlib
import matplotlib.pyplot as plt
import dg_model_ham as dg
import numpy as np
from numpy import linalg as la
from scipy.linalg import block_diag
import scipy
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

def rot(angle, vec):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    return(np.dot(R,vec))

def test_rot():
    v = np.array([ .480191,-1.614439])
    v1= np.array([-.480191,-1.614439])
    angles = np.linspace(0, np.pi, num=20)
    angles = angles[0:16] 
    for angle in angles:
        p = rot(angle,v)
        p1= rot(-angle,v1)
        plt.plot(p[0],p[1], 'r+')
        plt.plot(p1[0],p1[1],'b+')
        plt.xlim(-2,2)
        plt.ylim(-2,2)
    plt.show()

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

    ''' Computing H2O in x-y plane:
        bon angle remains fixed,
        O-H bonds get dissociated
    '''
    #test_rot()
    #exit()  

    log = open("log.txt", "w")
    f   = open("out.txt", "w")
    f.write("Computing H2O rotation in x-y-plane:\n")
    log.write("Computing H2O in x-y-plane:\n")
    
    start_pes = time.time()

    # Optimal geometry taken from cccbdb 
    #Mol  = [['H',[1.43, -.89, 0]], ['O',[0,0,0]], ['H',[-1.43,-.89,0 ]]]
    Mol  = [['H',[.480191,-1.614439, 0]], 
            ['O',[0,0,0]], 
            ['H',[-.480191,-1.614439,0 ]]]
    Mol1 = copy.deepcopy(Mol)

    angles = np.linspace(0, np.pi, num=20)#
    #angles = angles[0:2]
    angles = angles[0:16]

    bases = ['augccpvdz', '6311++g', '631++g', '321++g', 
             'ccpvdz'   , '6311g'  , '631g'  , '321g']
    acc   = [[.99, .95, .9, .85, .8, .7, .6, .5],
             [.99, .95, .9, .85, .8, .7, .6, .5],
             [.99, .95, .9, .85, .8, .7, .6, .5],
             [.99, .95, .9, .85, .8, .7, .6, .5],
             [.99, .95, .9, .85, .8, .7, .6, .5],
             [.99, .95, .9, .85, .8, .7, .6, .5],
             [.99, .95, .9, .85, .8, .7, .6, .5],
             [.99, .95, .9, .85, .8, .7, .6, .5],
             [.99, .95, .9, .85, .8, .7, .6, .5],
             [.99, .95, .9, .85, .8, .7, .6, .5]]

    #bases = ['sto3g', 'sto3g']
    #acc   = [[.4, .4], [.4, .4]]

    boxsize = [15] * 3
    dgrid   = [5]  * 3

    mfe     = np.zeros(len(angles))
    mfe_dg  = np.zeros(len(angles))
    mfe_vdg = np.zeros(len(angles))

    emp     = np.zeros(len(angles))
    emp_dg  = np.zeros(len(angles))
    emp_vdg = np.zeros(len(angles))
   
    ecc     = np.zeros(len(angles))
    ecc_dg  = np.zeros(len(angles))
    ecc_vdg = np.zeros(len(angles))

    mesh = [int(d * x) for d, x in zip(dgrid, boxsize)]

    for i, basis in enumerate(bases):
        f.write("AO basis:" + basis + "\n")
        log.write("AO basis:" + basis + "\n")
        f.write("Box size:\n")
        f.write(str(boxsize[0]) + " x " + str(boxsize[1]) + " x " + 
                str(boxsize[2]) + "\n")
        f.write("Box discretization:\n")
        f.write(str(mesh[0]) + " x " +str(mesh[1]) + " x " +
                str(mesh[2]) + "\n")

        for ac in acc[i]:
            f.write("SVD truncation:" + str(ac) + "\n")
            log.write("SVD truncation:" + str(ac) + "\n")

            for k, angle in enumerate(angles):

                Mol[0][1][0:2] = rot(angle, Mol1[0][1][0:2])
                Mol[2][1][0:2] = rot(-angle, Mol1[2][1][0:2])

                # centering molecule in the box:
                    
                # Centering Molecule in Box:
                ms, mm = mol_size(Mol)
                offset = np.array([ (bs - s)/2. - m for bs, s, m 
                                  in zip(boxsize, ms, mm)])

                for l, off in enumerate(offset):
                    for j in range(len(Mol)):
                        Mol[j][1][l] += off

                cell = gto.Cell()
                cell.a = [[boxsize[0], 0., 0.],
                          [0., boxsize[1], 0.],
                          [0., 0., boxsize[2]]]
                cell.unit    = 'B'
                cell.verbose = 3
                cell.basis   = basis #gth-dzvp, tzp
                cell.pseudo  = 'gth-pade'
                cell.mesh    = np.array(mesh)
                cell.atom    = Mol
                cell.build()

                # VDG calculations
                
                print("Creating  " + cell.basis +  "-VDG Hamiltonian ...")
                start_dg = time.time()
                
                cell_vdg  = dg.dg_model_ham(cell, None ,'rel_num', ac, True)
                
                end_dg   = time.time()
                print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
                print()

                # HF
                print("Computing HF in " + cell.basis +  "-VDG basis ...")
                start_hf   = time.time()
                
                try:
                    mfe_vdg[k] = cell_vdg.run_RHF()
                except:
                    mfe_vdg[k] = 1

                end_hf     = time.time()
                print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
                print()
         
                # MP2
                #print("Computing MP2 in " + cell.basis +  "-VDG basis ...")
                #start_mp = time.time()
                #
                #try:
                #    emp_vdg[k], _ = cell_vdg.run_MP2()
                #except:
                #    emp_vdg[k] = 1
                #
                #end_mp   = time.time()
                #print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
                #print()

                # CCSD
                #print("Computing CCSD in " + cell.basis +  "-VDG basis ...")
                #start_cc = time.time()
                #
                #try:
                #    ecc_vdg[k], _ = cell_vdg.run_CC()
                #except:
                #    ecc_vdg[k] = 1
                #
                #end_cc   = time.time()
                #print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
                #print()

                del cell_vdg
                # DG calculations
                
                print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
                start_dg = time.time()
                
                cell_dg  = dg.dg_model_ham(cell, None ,'rel_num', ac)
                
                end_dg   = time.time()
                print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
                print()
               
                # HF
                print("Computing HF in " + cell.basis +  "-DG basis ...")
                start_hf  = time.time()
                
                try:
                    mfe_dg[k] = cell_dg.run_RHF()
                except:
                    mfe_dg[k] = 1

                end_hf    = time.time()
                print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
                print()

                # MP2
                #print("Computing MP2 in " + cell.basis +  "-DG basis ...")
                #start_mp = time.time()
                # 
                #try:
                #    emp_dg[k], _ = cell_dg.run_MP2()
                #except:
                #    emp_dg[k] = 1
                #
                #end_mp   = time.time()
                #print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
                #print()

                # CCSD
                #print("Computing CCSD in " + cell.basis +  "-DG basis ...")
                #start_cc = time.time()    
                #
                #try:
                #    ecc_dg[k], _ = cell_dg.run_CC()
                #except:
                #    ecc_dg[k] = 1
                #
                #end_cc   = time.time()
                #print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
                #print()
            
                del cell_dg
                # Pure PySCF

                print("Computing HF in " + cell.basis +  " basis ...")
                start_hf = time.time()
                
                try:
                    mf = scf.RHF(cell, exxdiv='ewald') # madelung correction
                    mf.kernel(dump_chk=False)
                    mfe[k] = mf.e_tot
                except:
                    mfe[k] = 1

                end_hf = time.time()
                print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
                print()
                
                # MP2
                #print("Computing MP2 in " + cell.basis +  " basis ...")
                #start_mp = time.time()
                #
                #try:
                #    emp[k], _ = mp.MP2(mf).kernel() 
                #except:
                #    emp[k] = 1
                #
                #end_mp   = time.time()
                #print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
                #print()
                
                # CCSD    
                #print("Computing CCSD in " + cell.basis +  " basis ...")
                #start_cc = time.time()    
                #
                #try:
                #    cc_bi = cc.CCSD(mf)
                #    cc_bi.kernel()
                #    ecc[k] =cc_bi.e_corr
                #except:
                #    ecc[k] = 1 
                #
                #end_cc   = time.time()
                #print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
                #print()


            f.write("Rotation angle: "    + str(angles) + "\n")
            f.write("Mean-field energy: " + str(mfe)    + "\n")
            #f.write("MP2 energy: "        + str(emp)    + "\n")
            #f.write("CCSD energy: "       + str(ecc)    + "\n")

            f.write("Mean-field energy (DG): " + str(mfe_dg) + "\n")
            #f.write("MP2 energy (DG): "        + str(emp_dg) + "\n")
            #f.write("CCSD energy (DG): "       + str(ecc_dg) + "\n")

            f.write("Mean-field energy (VDG): " + str(mfe_vdg) + "\n")
            #f.write("MP2 energy (VDG): "        + str(emp_vdg) + "\n")
            #f.write("CCSD energy: "             + str(ecc_vdg) + "\n")
                
            print("Meanfield results:")
            print("  Builtin: ",mfe)
            print("  Minimum:", np.amin(mfe))
            print("  Maximum:", np.amax(mfe[5:-6]))
            print("  abs diff:", np.abs(np.amin(mfe)- np.amax(mfe[5:-6])))
            print("  DG: ", mfe_dg)
            print("  Minimum:", np.amin(mfe_dg))
            print("  Maximum:", np.amax(mfe_dg[5:-6]))
            print("  abs diff:", 
                    np.abs(np.amin(mfe_dg)- np.amax(mfe_dg[5:-6])))
    
            print("  VDG: ", mfe_vdg)
            print("  Minimum:", np.amin(mfe_vdg))
            print("  Maximum:", np.amax(mfe_vdg[5:-6]))
            print("  abs diff:", 
                    np.abs(np.amin(mfe_vdg)- np.amax(mfe_vdg[5:-6])))
    f.close()
    log.close()

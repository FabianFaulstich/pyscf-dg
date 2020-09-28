import sys
sys.path.append('../../../src')

import dg_model_ham as dg
import numpy as np
from numpy import linalg as la
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

def trans(x_, p_, a):
        '''
        counter clockwise a-rotation of p around x
        '''
        x, p  = np.array(x_), np.array(p_)
        c, s  = np.cos(a)[0], np.sin(a)[0]
        R     = np.array([[c, -s], [s, c]])
        vec   = p - x
        r_vec = np.dot(R,vec)
        out   = r_vec + x
        return out.tolist()

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

        log = open("log_lbox.txt", "w")
        f   = open("out_lbox.txt", "w")
        f.write("Computing L-box effect for HF and MP2:\n")
        log.write("Computing L-box effect for HF and MP2:\n")
        
        angle   = np.array([np.pi/3.0]) # pi/3
        shifts  = np.linspace(2, 8, num = 13) 

        bs    = [12] *3
        dgrid = [5]  *3 
        bases = ['augccpvdz', '6311++g', '631++g', '321++g', 
                    'ccpvdz', '6311g', '631g', '321g']
        accuracies = [.85, .7, .6, .7, 
                       .9, .7, .7, .8]
        #bases      = [ '321g', '321g']
        #accuracies = [ .99, .99] 

        Mol_init = [['H', [0,0,0]], ['H', [2,0,0]], 
                    ['H', [0,2,0]], ['H', [2,2,0]]]
        ms       = mol_size(Mol_init)
    
        mfe = np.zeros(len(shifts))
        mpe = np.zeros(len(shifts))
        
        mfe_dg = np.zeros(len(shifts))
        mpe_dg = np.zeros(len(shifts))

        # discretization mesh
        mesh = [int(d * x) for d, x in zip(dgrid, bs)]
        
        f.write("Box size:\n")
        f.write(str(bs) + " x " + str(bs) + " x " + str(bs) + "\n")
        f.write("Box discretization:\n")
        f.write(str(mesh[0]) + " x " +str(mesh[1]) + 
                " x " +str(mesh[2]) + "\n")
        log.write("Box size:\n")
        log.write(str(bs) + " x " + str(bs) + " x " + 
                  str(bs) + "\n")
        log.write("Box discretization:\n")
        log.write(str(mesh[0]) + " x " +str(mesh[1]) + 
                  " x " +str(mesh[2]) + "\n")

        for i, basis in enumerate(bases):
            
            log.write("    AO basis:\n")
            log.write("    " + basis + "\n")
            f.write("AO basis:" + basis + "\n")
            f.write("Rotation angle:" + str(angle) + "\n")
            
            acc = accuracies[i]

            f.write("Relative truncation in SVD:\n")
            f.write(str(acc) + "\n")
            log.write("    Relative truncation in SVD:\n")
            log.write("    " + str(acc) + "\n")
                
            for l, shift in enumerate(shifts):
           
                Mol = copy.deepcopy(Mol_init)
                Mol[2][1][0:2] = trans(Mol_init[0][1][0:2], 
                                       Mol_init[2][1][0:2], angle)
                Mol[3][1][0:2] = trans(Mol_init[1][1][0:2], 
                                       Mol_init[3][1][0:2], -angle)
                
                # Positioning Molecule in Box:
                ms, mm = mol_size(Mol)
                offset = np.array([ (d - s)/2. - m for d, s, m in zip(bs, 
                                  ms,mm)])
                offset[0] = shift
                for k, off in enumerate(offset):
                    for j in range(len(Mol)):
                        Mol[j][1][k] += off
                f.write("Shift along the X-coordinate:\n")
                f.write(str(shift -2) + "\n")
                #f.write("Molecule: \n")
                #f.write(str(Mol) +"\n")
                log.write("Shift along the X-coordinate:\n")
                log.write(str(shift -2) + "\n")
                
                cell         = gto.Cell()
                cell.a       = [[bs[0], 0., 0.], 
                                [0., bs[1], 0.], 
                                [0., 0., bs[2]]] 
                cell.unit    = 'B'
                cell.verbose = 3
                cell.basis   = basis
                cell.pseudo  = 'gth-pade'
                cell.mesh    = np.array(mesh)
                cell.atom    = Mol
                cell.build()
               
                f.write("Number of Built in basis functions: \n")
                f.write(str(cell.nao_nr()) + "\n")

                # VDG calculations
                print("Creating  VDG Hamiltonian ...")
                log.write("        Creating  VDG Hamiltonian ...\n")

                # Using vanilla voronoi
                cell_vdg = dg.dg_model_ham(cell, None ,'rel_num', acc, True)
                f.write("Number of DG basis functions:\n")
                f.write(str(cell_vdg.nao) + "\n")

                # HF in VDG
                log.write("        Computing HF in VDG Bases ...\n")
                start = time.time()
                
                mfe_dg[l] = cell_vdg.run_RHF()
                
                log.write("        Done! Elapsed time: " + 
                          str(time.time() - start) + "sec.\n")
            
                # MP2 in VDG
                log.write("        Computing MP2 in VDG Bases ...\n")
                start = time.time()
                
                mpe_dg[l], _ = cell_vdg.run_MP2()
                
                log.write("        Done! Elapsed time: " + 
                          str(time.time() - start) + "sec.\n")
                
                del cell_vdg
                
                # HF in builtin
                log.write("        Comuting HF using PyScf-PBC...\n")
                start = time.time()
                
                mf = scf.RHF(cell, exxdiv='ewald') # madelung correction
                mf.kernel(dump_chk = False)
                mfe[l] = mf.e_tot
                
                log.write("        Done! Elapsed time: " + 
                          str(time.time() - start) + "sec.\n")
                
                # MP2 in builtin
                log.write("        Comuting MP2 using PyScf-PBC...\n")
                start = time.time()
                
                mpe[l], _ = mp.MP2(mf).kernel()
                
                log.write("        Done! Elapsed time: " + 
                          str(time.time() - start) + "sec.\n")
                    
                del cell

            f.write("AO basis:\n")
            f.write(basis + "\n")
            f.write("Mean-field energy: " + str(mfe)    + "\n")
            f.write("MP2 corr. energy : " + str(mpe)    + "\n")
            
            f.write("Mean-field energy (DG): " + str(mfe_dg)    + "\n")
            f.write("MP2 corr. energy (DG) : " + str(mpe_dg)    + "\n")
    
            print("Mean-field energy: ", mfe)
            print()
            print("Mean-field energy (DG): ", mfe_dg)
            print()
        log.close()
        f.close()

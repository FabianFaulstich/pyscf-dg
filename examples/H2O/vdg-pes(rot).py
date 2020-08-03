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
    
if __name__ == '__main__':

    ''' Computing H2O in x-y plane:
        bon angle remains fixed,
        O-H bonds get dissociated
    '''
    #test_rot()
    #exit()  
    start_pes = time.time()

    # Optimal geometry taken from cccbdb 
    #Mol  = [['H',[1.43, -.89, 0]], ['O',[0,0,0]], ['H',[-1.43,-.89,0 ]]]
    Mol  = [['H',[.480191,-1.614439, 0]], ['O',[0,0,0]], ['H',[-.480191,-1.614439,0 ]]]
    Mol1 = copy.deepcopy(Mol)

    angles = np.linspace(0, np.pi, num=20)
    angles = angles[0:16]

    mfe     = np.zeros(len(angles))
    mfe_dg  = np.zeros(len(angles))
    mfe_vdg = np.zeros(len(angles))

    emp     = np.zeros(len(angles))
    emp_dg  = np.zeros(len(angles))
    emp_vdg = np.zeros(len(angles))
   
    ecc     = np.zeros(len(angles))
    ecc_dg  = np.zeros(len(angles))
    ecc_vdg = np.zeros(len(angles))
    

    for k, angle in enumerate(angles):
        Mol[0][1][0:2] = rot(angle, Mol1[0][1][0:2])
        Mol[2][1][0:2] = rot(-angle, Mol1[2][1][0:2])

        # Placing Molecule in the box (Oxigen in the center!) :

        offset    = np.array([10,10,3])
        atom_pos  = np.array([atom[1] for atom in Mol])
        atoms     = [atom[0] for atom in Mol]
        atom_off  = [5,5,1.5]
        atoms_box = np.array([pos + atom_off for pos in atom_pos])
        Mol_box   = [[atoms[i],atoms_box[i]] for i in range(len(atoms))]
        Mol_size = np.array([0,0,0])
        X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
        
        dgrid = [10]*3

        cell = gto.Cell()
        cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
        cell.unit    = 'B'
        cell.verbose = 3
        cell.basis   = 'tzp' #gth-dzvp, tzp
        cell.pseudo  = 'gth-pade'
        # potential speed up 
        #cell.ke_cutoff = 40.0
        cell.mesh    = np.array([int(d * x) for d, x in zip(dgrid, X)])
        cell.atom    = Mol_box
        cell.build()
        # Voronoi 2D:

        # 2D projection of atom position and grid:
        atoms_2d = np.array([atom[1][:2] for atom in Mol_box])
        mesh_2d  = np.unique(cell.get_uniform_grids()[:,:2], axis=0)

        # get Voronoi vertices + vertices on the boundary box (periodic voronoi):
        V_net = dg_tools.get_V_net_per(atoms_2d, np.amin(mesh_2d[:,0]), np.amax(mesh_2d[:,0]),
                            np.amin(mesh_2d[:,1]), np.amax(mesh_2d[:,1]))


        #voronoi_cells = dg_tools.get_V_cells(V_net, atoms_2d)
        voronoi_cells = None
        vert = np.array([elem[0] for elem in V_net])
        # get Voronoi cells:

        # DG vs VDG calculations
        print("Creating  " + cell.basis +  "-VDG Hamiltonian ...")
        start_dg = time.time()
        cell_vdg  = dg.dg_model_ham(cell, None ,'rel_num', 0.95, True, voronoi_cells, V_net)
        end_dg   = time.time()
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()

        print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
        start_dg = time.time()
        cell_dg  = dg.dg_model_ham(cell, None ,'rel_num', 0.95)
        end_dg   = time.time()
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()
        
        # HF
        print("Computing HF in " + cell.basis +  "-VDG basis ...")
        start_hf   = time.time()
        mfe_vdg[k] = cell_vdg.run_RHF()
        end_hf     = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

        print("Computing HF in " + cell.basis +  "-DG basis ...")
        start_hf  = time.time()
        mfe_dg[k] = cell_dg.run_RHF()
        end_hf    = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

        print("Computing HF in " + cell.basis +  " basis ...")
        start_hf = time.time()
        cell.verbose = 5
        mf = scf.RHF(cell, exxdiv='ewald') # madelung correction: ewlad
        mf.kernel(dump_chk=False)
        mfe[k] = mf.e_tot
        end_hf = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()
        
        # MP2
        print("Computing MP2 in " + cell.basis +  " basis ...")
        start_mp = time.time()
        #emp[i], _ = cell.run_MP2()
        end_mp   = time.time()
        print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
        print()

        print("Computing MP2 in " + cell.basis +  "-DG basis ...")
        start_mp = time.time()
        #emp_dg[i], _ = cell_dg.run_MP2()
        end_mp   = time.time()
        print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
        print()

        print("Computing MP2 in " + cell.basis +  "-VDG basis ...")
        start_mp = time.time()
        #emp_vdg[i], _ = cell_vdg.run_MP2()
        end_mp   = time.time()
        print("Done! Elapsed time: ", end_mp - start_mp, "sec.")
        print()

        # CCSD    
        print("Computing CCSD in " + cell.basis +  " basis ...")
        start_cc = time.time()    
        #ecc[i], _ = cell.run_CC()
        end_cc   = time.time()
        print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
        print()
        
        print("Computing CCSD in " + cell.basis +  "-DG basis ...")
        start_cc = time.time()    
        #ecc_dg[i], _ = cell_dg.run_CC()
        end_cc   = time.time()
        print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
        print()
    
        print("Computing CCSD in " + cell.basis +  "-VDG basis ...")
        start_cc = time.time()
        #ecc_vdg[i], _ = cell_vdg.run_CC()
        end_cc   = time.time()
        print("Done! Elapsed time: ", end_cc - start_cc, "sec.")
        print()


    print("Meanfield results:")
    print("  Builtin: ",mfe)
    print("  Minimum:", np.amin(mfe))
    print("  Maximum:", np.amax(mfe[5:-6]))
    print("  abs diff:", np.abs(np.amin(mfe) - np.amax(mfe[5:-6])))
    print("  DG: ", mfe_dg)
    print("  Minimum:", np.amin(mfe_dg))
    print("  Maximum:", np.amax(mfe_dg[5:-6]))
    print("  abs diff:", np.abs(np.amin(mfe_dg) - np.amax(mfe_dg[5:-6])))
    
    print("  VDG: ", mfe_vdg)
    print("  Minimum:", np.amin(mfe_vdg))
    print("  Maximum:", np.amax(mfe_vdg[5:-6]))
    print("  abs diff:", np.abs(np.amin(mfe_vdg) - np.amax(mfe_vdg[5:-6])))
    
    print()

    f = open("Output_SP.txt", "w")
    f.write("Meanfield results:\n")
    f.write(cell.basis+ ": "+ str(mfe) + "\n")
    f.write(cell.basis+ "-DG: " + str(mfe_dg)+ "\n")
    f.write(cell.basis+ "-VDG: "+ str(mfe_vdg)+ "\n")
    f.write("MP2 correlation energy:\n")
    f.write(cell.basis+": "+ str(emp)+ "\n")
    f.write(cell.basis+ "-DG: " + str(emp_dg)+ "\n")
    f.write(cell.basis+ "-VDG: "+ str(emp_vdg)+ "\n")
    f.write("CCSD correlation energy:\n")
    f.write(cell.basis+ ": " + str(ecc)+ "\n")
    f.write(cell.basis+ "-DG: " + str(ecc_dg)+ "\n")
    f.write(cell.basis+ "-VDG: " + str(ecc_vdg)+ "\n")
    f.close()

    #plt.plot(angles*180/np.pi, mfe   , 'b-v', label =  'HF  (' + cell.basis + ')')
    #plt.legend()
    #plt.show()

    #plt.plot(angles*180/np.pi, mfe_dg, 'r-v', label =  'HF  (' + cell.basis + '-DG)')
    #plt.plot(angles*180/np.pi, mfe_vdg, 'g--v', label =  'HF  (' + cell.basis + '-VDG)')
    #plt.legend()
    #plt.show()

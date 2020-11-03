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
    start_pes = time.time()

    # Optimal geometry taken from cccbdb 
    Mol  = [['H',[0,-1.6843386480164848, 0]], ['O',[0,0,0]], ['H',[0,-1.6843386480164848,0 ]]]
    Mol1 = copy.deepcopy(Mol)

    angles    = np.array([0.9117250946567979, np.pi/2])
    box_sizes = np.array([10,9,8]) 
    #box_sizes = np.array([3])
    basis = 'tzp'

    mfe     = np.zeros(len(angles))
    mfe_dg  = np.zeros(len(angles))
    mfe_vdg = np.zeros(len(angles))
    
    filename = "BS-calc_(" + basis + str(box_sizes) + ").txt" 
    f = open("BS-calc_DFT_("+str(box_sizes) + ").txt", "w")
    f.write("Computing poential barrier of H20 for different dis. boxes\n")
    f.write("Unterlying basis:\n")
    f.write(basis + "\n")

    for j, bs in enumerate(box_sizes):
        f.write("Box-size:\n")
        f.write(str(bs) + " x " + str(bs) + " x " + str(bs) + "\n")

        for k, angle in enumerate(angles):
    
            if k == 0:
                f.write("In optimal geometry, 104.5 deg\n")
            elif k == 1:
                f.write("On top of pes barrier, 180 deg\n")

            Mol[0][1][0:2] = rot(angle, Mol1[0][1][0:2])
            Mol[2][1][0:2] = rot(-angle, Mol1[2][1][0:2])

            # Placing Molecule in the box (Oxigen in the center!) :

            offset    = np.array([bs,bs,3])
            atom_pos  = np.array([atom[1] for atom in Mol])
            atoms     = [atom[0] for atom in Mol]
            atom_off  = [bs/2,bs/2,1.5]
            atoms_box = np.array([pos + atom_off for pos in atom_pos])
            Mol_box   = [[atoms[i],atoms_box[i]] for i in range(len(atoms))]
            Mol_size = np.array([0,0,0])
            X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
            
            dgrid = [7]*3

            cell = gto.Cell()
            cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
            cell.unit    = 'B'
            cell.verbose = 3
            cell.basis   = basis #gth-dzvp, tzp
            cell.pseudo  = 'gth-pade'
            cell.ke_cutoff = 40.0
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
            cell_vdg  = dg.dg_model_ham(cell, None ,'rel_num', 0.8, True, voronoi_cells, V_net)
            end_dg   = time.time()
            f.write("Elapsed time to create VDG-Ham: " + str(end_dg - start_dg) + "\n")
            print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
            print()

            print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
            start_dg = time.time()
            cell_dg  = dg.dg_model_ham(cell, None ,'rel_num', 0.8)
            end_dg   = time.time()
            f.write("Elapsed time to create DG-Ham: " + str(end_dg - start_dg) + "\n")
            print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
            print()
            
            # DFT
            print("Computing DFT in " + cell.basis +  "-VDG basis ...")
            start_hf   = time.time()
            mfe_vdg[k] = cell_vdg.run_DFT()
            end_hf     = time.time()
            f.write("Elapsed time to compute DFT in VDG: " + str(end_hf - start_hf)+ "\n")
            print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
            print()

            print("Computing DFT in " + cell.basis +  "-DG basis ...")
            start_hf  = time.time()
            mfe_dg[k] = cell_dg.run_DFT()
            end_hf    = time.time()
            f.write("Elapsed time to compute DFT in DG: " + str(end_hf - start_hf)+ "\n")
            print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
            print()

            print("Computing DFT in " + cell.basis +  " basis ...")
            start_hf = time.time()
            cell.verbose = 3
            mf = dft.RKS(cell)
            mf.xc = 'pbe'
            mf.kernel(dump_chk=False)
            mfe[k] = mf.e_tot
            end_hf = time.time()
            f.write("Elapsed time to compute DFT: " + str(end_hf - start_hf) + "\n")
            print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
            print()

        f.write("DFT results:\n")
        f.write("  Builtin: " + str(mfe) + "\n")
        f.write("  Minimum: " + str(np.amin(mfe))+ "\n")
        f.write("  Maximum: " + str(np.amax(mfe)) + "\n")
        f.write("  abs diff: " + str(np.abs(np.amin(mfe) - np.amax(mfe))) + "\n")
        
        f.write("  DG: " + str(mfe_dg) + "\n")
        f.write("  Minimum: " + str(np.amin(mfe_dg)) + "\n")
        f.write("  Maximum: " + str(np.amax(mfe_dg)) + "\n")
        f.write("  abs diff: " + str(np.abs(np.amin(mfe_dg) - np.amax(mfe_dg))) + "\n")
        
        f.write("  VDG: " + str(mfe_vdg) + "\n")
        f.write("  Minimum: " + str(np.amin(mfe_vdg)) + "\n")
        f.write("  Maximum: " + str(np.amax(mfe_vdg)) + "\n")
        f.write("  abs diff: " + str(np.abs(np.amin(mfe_vdg) - np.amax(mfe_vdg))) + "\n")

        print("DFT results:")
        print("  Builtin: ",mfe)
        print("  Minimum:", np.amin(mfe))
        print("  Maximum:", np.amax(mfe))
        print("  abs diff:", np.abs(np.amin(mfe) - np.amax(mfe)))
        
        print("  DG: ", mfe_dg)
        print("  Minimum:", np.amin(mfe_dg))
        print("  Maximum:", np.amax(mfe_dg))
        print("  abs diff:", np.abs(np.amin(mfe_dg) - np.amax(mfe_dg)))
        
        print("  VDG: ", mfe_vdg)
        print("  Minimum:", np.amin(mfe_vdg))
        print("  Maximum:", np.amax(mfe_vdg))
        print("  abs diff:", np.abs(np.amin(mfe_vdg) - np.amax(mfe_vdg)))
        

    f.close()


import sys
sys.path.append('../../../src')

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
        H-rotation pes barrier hight 
    '''
    start_pes = time.time()

    # Optimal geometry taken from cccbdb 
    Mol  = [['H',[0,-1.6843386480164848, 0]], ['O',[0,0,0]], ['H',[0,-1.6843386480164848,0 ]]]
    Mol1 = copy.deepcopy(Mol)

    angles    = np.array([0.9117250946567979, np.pi/2])
    box_sizes = np.array([9]) 
    #box_sizes = np.array([3])
    basis = ['qzp','tzp','dzp']
    accs  = [85, 99, 99]

    mfe     = np.zeros(len(angles))
    mfe_dg  = np.zeros(len(angles))
    mfe_vdg = np.zeros(len(angles))
    
    f = open("BS-calc_("+str(basis) +str(box_sizes) +").txt", "w")
    f.write("Computing potential barrier of H20 for different bases\n")
    f.write("Unterlying box size:\n")
    f.write(str(box_sizes[0]) + " x " + str(box_sizes[0]) + " x " + str(3) + "\n")

    for j, (bs, acc) in enumerate(zip(basis, accs)):
        print(j)
        print(bs)
        print(acc)
        f.write("Basis:\n")
        f.write(str(bs) +"\n")
        f.write("Kept singular values (relative number):\n")
        f.write(str(acc) + "\n")

        for k, angle in enumerate(angles):
            # k = 0: included angle 104.5 (optimal geometry)
            # k = 1: included angle 180   (on top of barrier)
            if k == 0:
                f.write("In optimal geometry, 104.5 deg\n")
            elif k == 1:
                f.write("On top of pes barrier, 180 deg\n")

            Mol[0][1][0:2] = rot(angle, Mol1[0][1][0:2])
            Mol[2][1][0:2] = rot(-angle, Mol1[2][1][0:2])

            # Placing Molecule in the box (Oxigen in the center!) :

            offset    = np.array([box_sizes[0], box_sizes[0], 3])
            atom_pos  = np.array([atom[1] for atom in Mol])
            atoms     = [atom[0] for atom in Mol]
            atom_off  = [box_sizes[0]/2,box_sizes[0]/2,1.5]
            atoms_box = np.array([pos + atom_off for pos in atom_pos])
            Mol_box   = [[atoms[i],atoms_box[i]] for i in range(len(atoms))]
            Mol_size = np.array([0,0,0])
            X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])
            dgrid = [8,8,7]

            cell = gto.Cell()
            cell.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
            cell.unit    = 'B'
            cell.verbose = 3
            cell.basis   = bs #gth-dzvp, tzp
            cell.pseudo  = 'gth-pade'
            cell.ke_cutoff = 40.0
            cell.mesh    = np.array([int(d * x) for d, x in zip(dgrid, X)])
            cell.atom    = Mol_box
            cell.build()
            f.write("Discretization: \n")
            f.write(str(cell.mesh) + "\n")
            # Voronoi 2D:

            # 2D projection of atom position and grid:
            atoms_2d = np.array([atom[1][:2] for atom in Mol_box])
            mesh_2d  = np.unique(cell.get_uniform_grids()[:,:2], axis=0)

            # get Voronoi vertices + vertices on the boundary box (periodic voronoi):
            V_net = dg_tools.get_V_net_per(atoms_2d, np.amin(mesh_2d[:,0]), np.amax(mesh_2d[:,0]),
                                np.amin(mesh_2d[:,1]), np.amax(mesh_2d[:,1]))
            
            voronoi_cells = dg_tools.get_V_cells(V_net, atoms_2d)
            #voronoi_cells = None
            vert = np.array([elem[0] for elem in V_net])
            # get Voronoi cells:


            # VDG calculations
            print("Creating  " + cell.basis +  "-VDG Hamiltonian ...")
            start_dg = time.time()
            cell_vdg  = dg.dg_model_ham(cell, None ,'rel_num', 0.85, True, voronoi_cells, V_net)
            end_dg   = time.time()
            f.write("Elapsed time to create VDG-Ham: " + str(end_dg - start_dg) + "\n")
            print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
            print()

            print("Computing HF in " + cell.basis +  "-VDG basis ...")
            start_hf   = time.time()
            mfe_vdg[k] = cell_vdg.run_RHF()
            end_hf     = time.time()
            f.write("Elapsed time to compute HF in VDG: " + str(end_hf - start_hf)+ "\n")
            print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
            print()
            del cell_vdg

            
            # DG calculations
            print("Creating  " + cell.basis +  "-DG Hamiltonian ...")
            start_dg = time.time()
            cell_dg  = dg.dg_model_ham(cell, None ,'rel_num', 0.85)
            end_dg   = time.time()
            f.write("Elapsed time to create DG-Ham: " + str(end_dg - start_dg) + "\n")
            print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
            print()
           
            print("Computing HF in " + cell.basis +  "-DG basis ...")
            start_hf  = time.time()
            mfe_dg[k] = cell_dg.run_RHF()
            end_hf    = time.time()
            f.write("Elapsed time to compute HF in DG: " + str(end_hf - start_hf)+ "\n")
            print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
            print()
            del cell_dg

            # Built-in calculcations
            print("Computing HF in " + cell.basis +  " basis ...")
            start_hf = time.time()
            cell.verbose = 3
            mf = scf.RHF(cell, exxdiv='ewald') # madelung correction: ewlad
            mf.kernel(dump_chk=False)
            mfe[k] = mf.e_tot
            end_hf = time.time()
            f.write("Elapsed time to compute HF: " + str(end_hf - start_hf) + "\n")
            print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
            print()
            del cell

        f.write("Meanfield results:\n")
        f.write("  Builtin: " + str(mfe) + "\n")
        f.write("  Minimum: " + str(mfe[0]) + "\n")
        f.write("  Maximum: " + str(mfe[1]) + "\n")
        f.write("  abs diff: " + str(mfe[0] - mfe[1]) + "\n")
        
        f.write("  DG: " + str(mfe_dg) + "\n")
        f.write("  Minimum: " + str(mfe_dg[0]) + "\n")
        f.write("  Maximum: " + str(mfe_dg[1]) + "\n")
        f.write("  abs diff: " + str(mfe_dg[0] - mfe_dg[1]) + "\n")
        
        f.write("  VDG: " + str(mfe_vdg) + "\n")
        f.write("  Minimum: " + str(mfe_vdg[0]) + "\n")
        f.write("  Maximum: " + str(mfe_vdg[1]) + "\n")
        f.write("  abs diff: " + str(mfe_vdg[0] - mfe_vdg[1]) + "\n")

        print("Meanfield reangles:")
        print("  Builtin: ",mfe)
        print("  Minimum:", mfe[0])
        print("  Maximum:", mfe[1])
        print("  abs diff:", mfe[0] - mfe[1])
        
        print("  DG: ", mfe_dg)
        print("  Minimum:", mfe_dg[0])
        print("  Maximum:", mfe_dg[1])
        print("  abs diff:", mfe_dg[0] - mfe_dg[1])
        
        print("  VDG: ", mfe_vdg)
        print("  Minimum:", mfe_vdg[0])
        print("  Maximum:", mfe_vdg[1])
        print("  abs diff:", mfe_vdg[0] - mfe_vdg[1])
        
    f.close()

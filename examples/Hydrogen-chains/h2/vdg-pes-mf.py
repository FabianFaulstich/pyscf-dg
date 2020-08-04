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

from scipy.optimize import curve_fit

def func(x,Y,a):
    return Y + a * x**(-3)

if __name__ == '__main__':

    start_pes = time.time()

    dgrid = [4]*3
    bonds = np.array([[12e-1], [14e-1], [16e-1], [18e-1], [20e-1]])
    #bonds = np.array([[8e-1], [10e-1], [12e-1], [14e-1], [16e-1], [18e-1], [20e-1], [24e-1], [28e-1], [32e-1], [36e-1]])
    #bonds = np.array([[8e-1]])
    bonds_max = np.max(bonds)
    offset = np.array([10., 6., 6.])
    X = np.array([int(np.ceil(offset[0] + bonds_max)), offset[1], offset[2]])

    cell_dzp = gto.Cell()
    cell_tzp = gto.Cell()
    cell_qzp = gto.Cell()

    atom_spec = 'H'
    
    cell_dzp.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
    cell_tzp.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
    cell_qzp.a = [[X[0],0.,0.],[0.,X[1],0],[0,0,X[2]]]
    
    cell_dzp.unit = 'B'
    cell_dzp.verbose = 3
    cell_dzp.basis = 'dzp'
    cell_dzp.pseudo = 'gth-pade'
    cell_dzp.mesh = np.array([int(d * x) for d, x in zip(dgrid, X)])
    
    cell_tzp.unit = 'B'
    cell_tzp.verbose = 3
    cell_tzp.basis = 'tzp'
    cell_tzp.pseudo = 'gth-pade'
    cell_tzp.mesh = np.array([int(d * x) for d, x in zip(dgrid, X)])

    cell_qzp.unit = 'B'
    cell_qzp.verbose = 3
    cell_qzp.basis = 'qzp'
    cell_qzp.pseudo = 'gth-pade'
    cell_qzp.mesh = np.array([int(d * x) for d, x in zip(dgrid, X)])

    mfe_dzp     = np.zeros(len(bonds))
    mfe_dg_dzp  = np.zeros(len(bonds))
    mfe_vdg_dzp = np.zeros(len(bonds))

    mfe_tzp     = np.zeros(len(bonds))
    mfe_dg_tzp  = np.zeros(len(bonds))
    mfe_vdg_tzp = np.zeros(len(bonds))

    mfe_qzp     = np.zeros(len(bonds))
    mfe_dg_qzp  = np.zeros(len(bonds))
    mfe_vdg_qzp = np.zeros(len(bonds))

    mfe_cbs = np.zeros(len(bonds))

    for i, bd in enumerate(bonds):
        # atom position in z-direction
        Z = (X[0]-np.sum(bd))/2.0 + np.append( 0, np.cumsum(bd))
        cell_dzp.atom = [[atom_spec, (Z[0], X[1]/2, X[2]/2)],
                     [atom_spec, (Z[1], X[1]/2, X[2]/2)]]
        cell_dzp.build()
        cell_tzp.atom = [[atom_spec, (Z[0], X[1]/2, X[2]/2)],
                     [atom_spec, (Z[1], X[1]/2, X[2]/2)]]
        cell_tzp.build()
        cell_qzp.atom = [[atom_spec, (Z[0], X[1]/2, X[2]/2)],
                     [atom_spec, (Z[1], X[1]/2, X[2]/2)]]
        cell_qzp.build()

        # DG vs VDG calculations
        print("Creating  VDG Hamiltonianis ...")
        start_dg = time.time()
        atoms_2d = np.array([atom[1][:2] for atom in cell_dzp.atom])
        V_net = dg_tools.get_V_net_per(atoms_2d, 0, cell_dzp.a[0][0],0, cell_dzp.a[1][1])
        voronoi_cells = dg_tools.get_V_cells(V_net, atoms_2d)
        #cell_vdg_dzp  = dg.dg_model_ham(cell_dzp, None ,'rel_num', 0.99, True, voronoi_cells)
        #cell_vdg_tzp  = dg.dg_model_ham(cell_tzp, None ,'rel_num', 0.99, True, voronoi_cells)
        cell_vdg_qzp  = dg.dg_model_ham(cell_qzp, None ,'rel_num', 0.99, True, voronoi_cells)
        end_dg   = time.time()
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()

        # DG calculations
        print("Creating DG Hamiltonians ...")
        start_dg = time.time()
        #cell_dg_dzp  = dg.dg_model_ham(cell_dzp) # default setting dg_trunc = 'abs_tol', svd_tol = 1e-3
        #cell_dg_tzp  = dg.dg_model_ham(cell_tzp)
        cell_dg_qzp  = dg.dg_model_ham(cell_qzp)
        end_dg   = time.time() 
        print("Done! Elapsed time: ", end_dg - start_dg, "sec.")
        print()

        # HF
        print("Computing HF in DG bases ...")
        start_hf = time.time()
        #mfe_dg_dzp[i] = cell_dg_dzp.run_RHF()
        #mfe_dg_tzp[i] = cell_dg_tzp.run_RHF()
        mfe_dg_qzp[i] = cell_dg_qzp.run_RHF()
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

        print("Computing HF in VDG basis ...")
        start_hf = time.time()
        #mfe_vdg_dzp[i] = cell_vdg_dzp.run_RHF()
        #mfe_vdg_tzp[i] = cell_vdg_tzp.run_RHF()
        mfe_vdg_qzp[i] = cell_vdg_qzp.run_RHF()
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

        print("Computing HF ...")
        start_hf = time.time()
        mf_dzp = scf.RHF(cell_dzp, exxdiv='ewald') # madelung correction
        mf_dzp.kernel()
        mfe_dzp[i] = mf_dzp.e_tot
        mf_tzp = scf.RHF(cell_tzp, exxdiv='ewald') # madelung correction
        mf_tzp.kernel()
        mfe_tzp[i] = mf_tzp.e_tot
        mf_qzp = scf.RHF(cell_qzp, exxdiv='ewald') # madelung correction
        mf_qzp.kernel()
        mfe_qzp[i] = mf_qzp.e_tot
        end_hf   = time.time()
        print("Done! Elapsed time: ", end_hf - start_hf, "sec.")
        print()

    end_pes = time.time()
    print("Elapsed time: ", end_pes - start_pes, "sec.")

    X = np.array([2,3,4])
    for i in range(len(mfe_cbs)):
        s = np.array([mfe_dzp[i],mfe_tzp[i],mfe_qzp[i]])
        popt, pcov = curve_fit(func, X, s)
        print(popt[0])
        mfe_cbs[i] = popt[0]


    print("Meanfield results:")
    print("Builtin: ",mfe_dzp)
    print("DG: ", mfe_dg_dzp)
    print("VDG: ", mfe_vdg_dzp)
    print()
    print("Builtin: ",mfe_tzp)
    print("DG: ", mfe_dg_tzp)
    print("VDG: ", mfe_vdg_tzp)
    print()
    print("Builtin: ",mfe_qzp)
    print("DG: ", mfe_dg_qzp)
    print("VDG: ", mfe_vdg_qzp)
    print()

    np.savetxt('Energies_vdg_mf.txt',(mfe_dzp,mfe_dg_dzp,mfe_vdg_dzp,mfe_tzp,mfe_dg_tzp,mfe_vdg_tzp,mfe_qzp,mfe_dg_qzp,mfe_vdg_qzp))


    bonds_plt = [bd[0] for bd in bonds]
    #plt.plot(bonds_plt, mfe_dzp   , 'b-v', label =  'HF  (DZP)')
    #plt.plot(bonds_plt, mfe_dg_dzp, 'r-v', label =  'HF  (DZP)')
    #plt.plot(bonds_plt, mfe_vdg_dzp, 'g--v', label =  'HF  (DZP)')
    #plt.plot(bonds_plt, mfe_tzp   , 'b->', label =  'HF  (TZP)')
    #plt.plot(bonds_plt, mfe_dg_tzp, 'r->', label =  'HF  (TZP)')
    #plt.plot(bonds_plt, mfe_vdg_tzp, 'g-->', label =  'HF  (TZP)')
    plt.plot(bonds_plt, mfe_qzp   , 'b-^', label =  'HF  (QZP)')
    plt.plot(bonds_plt, mfe_dg_qzp, 'r-^', label =  'HF  (QZP)')
    plt.plot(bonds_plt, mfe_vdg_qzp, 'g--^', label =  'HF  (QZP)')
    plt.plot(bonds_plt, mfe_cbs, 'm-*', label =  'HF  (CBS)')

    plt.legend()
    plt.show()


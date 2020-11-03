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

    log = open("log.txt", "w")
    f   = open("energies.txt", "w")
    box_sizes = np.array([10., 9., 8.])

    dgrid = [8,8,8] # 
    #bonds = np.array([[12e-1], [14e-1], [16e-1], [18e-1], [20e-1]])
    #bonds = np.array([[8e-1], [10e-1], [12e-1], [14e-1], [16e-1], [18e-1], 
    #                 [20e-1], [24e-1], [28e-1], [32e-1], [36e-1]])
    bonds = np.array([[8e-1]])
    bonds_max = np.max(bonds)
   
    for j, bs in enumerate(box_sizes):

        start_pes = time.time()
        
        offset = np.array([bs, bs, bs])
        X = np.array([int(np.ceil(offset[0] + bonds_max)), offset[1], offset[2]])
        mesh = [int(d * x) for d, x in zip(dgrid, X)]

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
        cell_dzp.mesh = np.array(mesh)
        
        cell_tzp.unit = 'B'
        cell_tzp.verbose = 3
        cell_tzp.basis = 'tzp'
        cell_tzp.pseudo = 'gth-pade'
        cell_tzp.mesh = np.array(mesh)

        cell_qzp.unit = 'B'
        cell_qzp.verbose = 3
        cell_qzp.basis = 'qzp'
        cell_qzp.pseudo = 'gth-pade'
        cell_qzp.mesh = np.array(mesh)

        log.write("Box size:\n")
        log.write(str(X[0]) + " x " + str(X[1]) + " x " +str(X[2]) + "\n")
        log.write("Box discretization:\n")
        log.write(str(mesh[0]) + " x " + str(mesh[1]) + " x " + \
                str(mesh[2]) + "\n")
        f.write("Box size:\n")
        f.write(str(X[0]) + " x " + str(X[1]) + " x " +str(X[2]) + "\n")
        f.write("Box discretization:\n")
        f.write(str(mesh[0]) + " x " + str(mesh[1]) + " x " +str(mesh[2]) + "\n")


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
            log.write("    Computing PES:\n")
            f.write("    Bondlength: " + str(bd) + "\n")
            # atom position in x-direction
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

            # VDG calculations
            print("Creating  VDG Hamiltonianis ...")
            log.write("    Creating  VDG Hamiltonianis ...\n")
            start_H = time.time()
            
            log.write("        QZP-VDG Hamiltonian ... \n")
            start = time.time()
            
            cell_vdg_qzp  = dg.dg_model_ham(cell_qzp, dg_cuts = None, 
                    dg_trunc = 'rel_num', svd_tol = 0.99, voronoi = True,
                    dg_on=True, gram = None)
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")

            log.write("        DZP-VDG Hamiltonian ... \n")
            start = time.time()
            
            cell_vdg_dzp  = dg.dg_model_ham(cell_dzp, dg_cuts = None,
                    dg_trunc = 'rel_num', svd_tol = 0.99, voronoi = True,
                    dg_on=True, gram = None) 
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            log.write("        TZP-VDG Hamiltonian ... \n")
            start = time.time()
            
            cell_vdg_tzp  = dg.dg_model_ham(celli_tzp, dg_cuts = None,
                    dg_trunc = 'rel_num', svd_tol = 0.99, voronoi = True,
                    dg_on=True, gram = None)
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            end_H = time.time()
            print("Done! Elapsed time: ", end_H - start_H, "sec.")
            log.write("    Done! Elapsed time: " + str(end_H - start_H) \
                    + "sec.\n")
            print()

            # HF in DG-V
            log.write("    Computing HF in DG-V Bases ...\n")
            start_H = time.time()

            log.write("        in DG-V-DZP ... \n")
            start = time.time()

            mfe_vdg_dzp[i] = cell_vdg_dzp.run_RHF()
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            log.write("        in TZP-VDG ... \n")
            start = time.time()
            
            mfe_vdg_tzp[i] = cell_vdg_tzp.run_RHF()
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            log.write("        in QZP-VDG ... \n")
            start = time.time()
            
            mfe_vdg_qzp[i] = cell_vdg_qzp.run_RHF()
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            end_H = time.time()
            print("Done! Elapsed time: ", end_H - start_H, "sec.")
            log.write("    Done! Elapsed time: " + str(end_H - start_H) \
                    + "sec.\n")
            print()

            del cell_vdg_dzp        
            del cell_vdg_tzp
            del cell_vdg_qzp

            # DG-R calculations
            print("Creating  DG-R Hamiltonianis ...")
            log.write("    Creating DG-R Hamiltonianis ...\n")
            start_H = time.time()

            log.write("        DG-R-DZP Hamiltonian ... \n")
            start = time.time()
            
            cell_dg_dzp  = dg.dg_model_ham(cell_dzp)
            #, None ,'rel_num', 0.99, True, voronoi_cells)
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            log.write("        DG-R-TZP Hamiltonian ... \n")
            start = time.time()

            cell_dg_tzp  = dg.dg_model_ham(cell_tzp)
            #, None ,'rel_num', 0.99, True, voronoi_cells)
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            log.write("        DG-R-QZP Hamiltonian ... \n")
            start = time.time()

            cell_dg_qzp  = dg.dg_model_ham(cell_qzp)
            #, None ,'rel_num', 0.99, True, voronoi_cells)
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
           
            end_H = time.time()
            print("Done! Elapsed time: ", end_H - start_H, "sec.")
            log.write("    Done! Elapsed time: " + str(end_H - start_H) \
                    + "sec.\n")
            print()

            # HF in DG-R
            print("Computing HF ...")
            log.write("    Computing HF in DG Bases ...\n")
            start_H = time.time()
            log.write("        in DZP-DG ... \n")
            start = time.time()
            
            mfe_dg_dzp[i] = cell_dg_dzp.run_RHF()
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
           
            log.write("        in TZP-DG ... \n")
            start = time.time()
            
            mfe_dg_tzp[i] = cell_dg_tzp.run_RHF()
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            log.write("        in QZP-DG ... \n")
            start = time.time()
            
            mfe_dg_qzp[i] = cell_dg_qzp.run_RHF()
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            end_H = time.time()
            print("Done! Elapsed time: ", end_H - start_H, "sec.")
            log.write("    Done! Elapsed time: " + str(end_H - start_H) \
                    + "sec.\n")
        
            del cell_dg_dzp
            del cell_dg_tzp
            del cell_dg_qzp
            
            # HF in builtin
            log.write("    Comuting HF in regular Bases ...\n")
            start_H = time.time()

            log.write("        in DZP ... \n")
            start = time.time()
            
            mf_dzp = scf.RHF(cell_dzp, exxdiv='ewald') # madelung correction
            mf_dzp.kernel()
            mfe_dzp[i] = mf_dzp.e_tot           
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            log.write("        in TZP ... \n")
            start = time.time()
            
            mf_tzp = scf.RHF(cell_tzp, exxdiv='ewald') # madelung correction
            mf_tzp.kernel()
            mfe_tzp[i] = mf_tzp.e_tot
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            log.write("        in QZP ... \n")
            start = time.time()        
            
            mf_qzp = scf.RHF(cell_qzp, exxdiv='ewald') # madelung correction
            mf_qzp.kernel()
            mfe_qzp[i] = mf_qzp.e_tot
            
            log.write("        Done! Elapsed time: " + str(time.time() - start) \
                    + "sec.\n")
            end_H = time.time()
            print("Done! Elapsed time: ", end_H - start_H, "sec.")
            log.write("    Done! Elapsed time: " + str(end_H - start_H) + "sec.\n")
            print()

        f.write("    Meanfield results:\n")
        f.write("    DZP     : " + str(mfe_dzp)     + "\n")
        f.write("    DZP-DG  : " + str(mfe_dg_dzp)  + "\n")
        f.write("    DZP-VDG : " + str(mfe_vdg_dzp) + "\n")
        f.write("    TZP     : " + str(mfe_tzp)     + "\n")
        f.write("    TZP-DG  : " + str(mfe_dg_tzp)  + "\n")
        f.write("    TZP-VDG : " + str(mfe_vdg_tzp) + "\n")
        f.write("    QZP     : " + str(mfe_qzp)     + "\n")
        f.write("    QZP-DG  : " + str(mfe_dg_qzp)  + "\n")
        f.write("    QZP-VDG : " + str(mfe_vdg_qzp) + "\n")
    f.close()
    log.close()
        
    
        #X_cbs = np.array([2,3,4])
        #for i in range(len(mfe_cbs)):
        #    s = np.array([mfe_dzp[i],mfe_tzp[i],mfe_qzp[i]])
        #    s = np.array([-1.02133, -1.04015, -1.04259])
        #    print(X_cbs)
        #    print(s)
        #    popt, pcov = curve_fit(func, X_cbs, s)
        #    print(popt[0])
        #    mfe_cbs[i] = popt[0]
            
        #f.write("    CBS-Lim : " + str(mfe_cbs)     + "\n")

    #bonds_plt = [bd[0] for bd in bonds]
    #plt.plot(bonds_plt, mfe_dzp   , 'b-v', label =  'HF  (DZP)')
    #plt.plot(bonds_plt, mfe_dg_dzp, 'r-v', label =  'HF  (DZP)')
    #plt.plot(bonds_plt, mfe_vdg_dzp, 'g--v', label =  'HF  (DZP)')
    #plt.plot(bonds_plt, mfe_tzp   , 'b->', label =  'HF  (TZP)')
    #plt.plot(bonds_plt, mfe_dg_tzp, 'r->', label =  'HF  (TZP)')
    #plt.plot(bonds_plt, mfe_vdg_tzp, 'g-->', label =  'HF  (TZP)')
    #plt.plot(bonds_plt, mfe_qzp   , 'b-^', label =  'HF  (QZP)')
    #plt.plot(bonds_plt, mfe_dg_qzp, 'r-^', label =  'HF  (QZP)')
    #plt.plot(bonds_plt, mfe_vdg_qzp, 'g--^', label =  'HF  (QZP)')
    #plt.plot(bonds_plt, mfe_cbs, 'm-*', label =  'HF  (CBS)')

    #plt.legend()
    #plt.show()


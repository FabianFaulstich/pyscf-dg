import sys
sys.path.append('../../src')

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


from pyscf.pbc.tools import pyscf_ase
from ase.io import read, write

def graphene_ase():
    
    supercell = read('graphene.xyz')
    # FIXME when you compare with supercell.arrays['positions'], you see
    # that the difference with the array below is just the last few
    # digits. Somehow this leads the Voronoi procedure to complain. Maybe there is some bug
    supercell.arrays['positions'] = np.array([[0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00],
        [2.673768765258395e-16, 1.414508159514583e+00, 0.000000000000000e+00],
        [1.225000000000000e+00, 2.121762239271875e+00, 0.000000000000000e+00],
        [1.225000000000000e+00, 3.536270398786459e+00, 0.000000000000000e+00]])

    print('Convert to pyscf')
    
    cell = gto.Cell()
    cell.verbose = 5
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(supercell)
    cell.a=supercell.cell


    return cell


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

        cell = graphene_ase()

        log = open("log.txt", "w")
        f   = open("out.txt", "w")
        #test_trans()
        #angles = np.linspace(0, np.pi/2.0, num=7)
        
        lat = cell.lattice_vectors() 
        boxsize = np.diag(lat)
        dgrid    = [6] * 3 
        #bases    = ['augccpvdz', '6311++g', '631++g', '321++g', '6311+g', '631+g', 'ccpvdz', '6311g', '631g', '321g']
        bases    = [ 'gth-dzvp', 'gth-dzvp']
        #acc      = [[.99, .95, .9, .85, .8, .7, .6, .5],
        #            [.99, .95, .9, .85, .8, .7, .6, .5],
        #            [.99, .95, .9, .85, .8, .7, .6, .5],
        #            [.99, .95, .9, .85, .8, .7, .6, .5],
        #            [.99, .95, .9, .85, .8, .7, .6, .5],
        #            [.99, .95, .9, .85, .8, .7, .6, .5],
        #            [.99, .95, .9, .85, .8, .7, .6, .5],
        #            [.99, .95, .9, .85, .8, .7, .6, .5],
        #            [.99, .95, .9, .85, .8, .7, .6, .5],
        #            [.99, .95, .9, .85, .8, .7, .6, .5]]
        acc = [[.99], [.99]] 

    
        #mfe = np.zeros(len(angles))
        #max_ev = np.zeros(len(angles))
        #min_ev = np.zeros(len(angles))
        #con_no = np.zeros(len(angles))
        #m      = np.zeros(len(angles)) 
        
        #mfe_dg = np.zeros(len(angles))
        #max_ev_dg = np.zeros(len(angles))
        #min_ev_dg = np.zeros(len(angles))
        #con_no_dg = np.zeros(len(angles))
        #m_dg      = np.zeros(len(angles)) 

        mesh = [int(d * x) for d, x in zip(dgrid, boxsize)]
        
        f.write("Box size:\n")
        f.write(str(boxsize[0]) + " x " + str(boxsize[1]) + " x " + str(boxsize[2]) + "\n")
        f.write("Box discretization:\n")
        f.write(str(mesh[0]) + " x " +str(mesh[1]) + " x " +str(mesh[2]) + "\n")
        log.write("Box size:\n")
        log.write(str(boxsize[0]) + " x " + str(boxsize[1]) + " x " + str(boxsize[2]) + "\n")
        log.write("Box discretization:\n")
        log.write(str(mesh[0]) + " x " +str(mesh[1]) + " x " +str(mesh[2]) + "\n")

        for ac, basis in enumerate(bases):
            
            f.write("AO basis:\n")
            f.write(basis + "\n")
            log.write("    AO basis:\n")
            log.write("    " + basis + "\n")
            
            for accc in acc[ac]:
                f.write("Relative truncation in SVD:\n")
                f.write(str(accc) + "\n")
                log.write("    Relative truncation in SVD:\n")
                log.write("    " + str(accc) + "\n")
                
               
                
                cell.basis   = basis
                cell.pseudo  = 'gth-pade'
                cell.mesh    = np.array(mesh)
                cell.build()

                overlap = cell.pbc_intor('int1e_ovlp_sph')
                w, _ = la.eig(overlap)
                max_ev = np.amax(w)
                min_ev = np.amin(w)
                con_no = np.amax(w)/ np.amin(w)
                m = len(w[w<10**-5])
                
                print("Max eigenvalue: "   , max_ev )
                print("Min eigenvalue: "   , min_ev )
                print("Condition no.: "    , con_no )
                print("No. of EV<10e-5: "  , m )

                # VDG calculations
                print("Creating  VDG Hamiltonian ...")
                log.write("        Creating  VDG Hamiltonian ...\n")
                start_H = time.time()

                log.write("            Creating Voronoi decomposition ... \n")
                start = time.time()
                
                # pay attention to the unit
                atoms_2d = cell.atom_coords()[:,:2]
                V_net = dg_tools.get_V_net_per(atoms_2d, 0, cell.lattice_vectors()[0,0],0, 
                        cell.lattice_vectors()[1,1])
                voronoi_cells = dg_tools.get_V_cells(V_net, atoms_2d)
                

                log.write("            Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                log.write("            VDG Hamiltonian ... \n")
                start = time.time()
                
                cell_vdg = dg.dg_model_ham(cell, None ,'rel_num', accc, True, voronoi_cells, V_net)
                f.write("Number of DG basis functions:\n")
                f.write(str(cell.nao) + "\n")
                log.write("            Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                log.write("        Done! Elapsed time: " + str(time.time() - start_H) + "sec.\n")

                overlap_dg = cell_vdg.ovl_dg
                w_dg, _ = la.eig(overlap_dg)
                max_ev_dg  = np.amax(w_dg)
                min_ev_dg  = np.amin(w_dg)
                con_no_dg  = np.amax(w_dg)/ np.amin(w_dg)
                m_dg  = len(w_dg[w_dg<10**-5])
                
                print("Max eigenvalue (DG): "   , max_ev_dg )
                print("Min eigenvalue (DG): "   , min_ev_dg )
                print("Condition no. (DG): "    , con_no_dg )
                print("No. of EV<10e-5 (DG): "  , m_dg )


                # HF in VDG
                log.write("        Computing HF in VDG Bases ...\n")
                start = time.time()
                
                mfe_dg  = cell_vdg.run_RHF()
                
                log.write("        Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
            

                # HF in builtin
                log.write("        Comuting HF using PyScf-PBC...\n")
                start = time.time()
                
                mf = scf.RHF(cell, exxdiv='ewald') # madelung correction
                mf.kernel(dump_chk = False)
                mfe  = mf.e_tot
                
                log.write("        Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                
                del cell
                del cell_vdg


                f.write("Mean-field energy: " + str(mfe) + "\n")
                f.write("Max eigenvalue: "    + str(max_ev) + "\n") 
                f.write("Min eigenvalue: "    + str(min_ev) + "\n")
                f.write("Condition no.: "     + str(con_no) + "\n")
                f.write("No. of EV<10e-5: "   + str(m) + "\n")
                
                f.write("Mean-field energy (DG): " + str(mfe_dg) + "\n")
                f.write("Max eigenvalue (DG): "    + str(max_ev_dg) + "\n")
                f.write("Min eigenvalue (DG): "    + str(min_ev_dg) + "\n")
                f.write("Condition no. (DG): "     + str(con_no_dg) + "\n")
                f.write("No. of EV<10e-5 (DG): "   + str(m_dg) + "\n")

        
                print("Mean-field energy: ", mfe)
                print("Max eigenvalue: "   , max_ev)
                print("Min eigenvalue: "   , min_ev)
                print("Condition no.: "    , con_no)
                print("No. of EV<10e-5: "  , m)
                print()
                print("Mean-field energy (DG): ", mfe_dg)
                print("Max eigenvalue (DG): "   , max_ev_dg)
                print("Min eigenvalue (DG): "   , min_ev_dg)
                print("Condition no. (DG): "    , con_no_dg)
                print("No. of EV<10e-5 (DG): "  , m_dg)
                print()
        log.close()
        f.close()

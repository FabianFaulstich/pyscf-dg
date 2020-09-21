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
        x, p = np.array(x_), np.array(p_)
        c, s  = np.cos(a), np.sin(a)
        R     = np.array(((c, -s), (s, c)))
        vec   = p - x
        r_vec = np.dot(R,vec)
        out = r_vec + x
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

        log = open("log.txt", "w")
        f   = open("out.txt", "w")
        f.write("Computing P4 to D4 transition correlation energies:\n")
        log.write("Computing P4 to D4 transition correlation energies:\n")
        #test_trans()
        angles = np.linspace(0, np.pi/2.0, num=7)
        angles = angles[1:]

        boxsizes = np.array([12])
        dgrid    = [5] * 3 
        bases    = ['augccpvdz', '6311++g', '631++g', '321++g', '6311+g', '631+g', 'ccpvdz', '6311g', '631g', '321g']
        #bases    = [ '321g']
        acc      = [[.85], [.7], [.6], [.7], [.7], [.7], [.9], [.7], [.7], [.8]]
        #acc = [[.5]] 


        Mol_init = [['H', [0,0,0]],['H', [2,0,0]], ['H', [0,2,0]], ['H', [2,2,0]]]
        ms = mol_size(Mol_init)
    
        mfe = np.zeros(len(angles))
        mpe = np.zeros(len(angles))
        cce = np.zeros(len(angles))
        max_ev = np.zeros(len(angles))
        min_ev = np.zeros(len(angles))
        con_no = np.zeros(len(angles))
        m      = np.zeros(len(angles)) 
        
        mfe_dg = np.zeros(len(angles))
        mpe_dg = np.zeros(len(angles))
        cce_dg = np.zeros(len(angles))
        max_ev_dg = np.zeros(len(angles))
        min_ev_dg = np.zeros(len(angles))
        con_no_dg = np.zeros(len(angles))
        m_dg      = np.zeros(len(angles)) 

        for bs in boxsizes:
            
            # discretization mesh
            mesh = [int(d * x) for d, x in zip(dgrid, [bs]*3)]
            
            f.write("Box size:\n")
            f.write(str(bs) + " x " + str(bs) + " x " + str(bs) + "\n")
            f.write("Box discretization:\n")
            f.write(str(mesh[0]) + " x " +str(mesh[1]) + " x " +str(mesh[2]) + "\n")
            log.write("Box size:\n")
            log.write(str(bs) + " x " + str(bs) + " x " + str(bs) + "\n")
            log.write("Box discretization:\n")
            log.write(str(mesh[0]) + " x " +str(mesh[1]) + " x " +str(mesh[2]) + "\n")

            for ac, basis in enumerate(bases):
                
                f.write("AO basis:\n")
                f.write(basis + "\n")
                log.write("    AO basis:\n")
                log.write("    " + basis + "\n")
                f.write("Rotation angles:\n")
                f.write(str(angles) + "\n")
                
                for accc in acc[ac]:
                    f.write("Relative truncation in SVD:\n")
                    f.write(str(accc) + "\n")
                    log.write("    Relative truncation in SVD:\n")
                    log.write("    " + str(accc) + "\n")
                    
                    for i, a in enumerate(angles):
                        
                        log.write("        Rotation angle:\n")
                        log.write("        " + str(a) + "\n") 
                       
                        Mol = copy.deepcopy(Mol_init)
                        Mol[2][1][0:2] = trans(Mol_init[0][1][0:2], Mol_init[2][1][0:2], a)
                        Mol[3][1][0:2] = trans(Mol_init[1][1][0:2], Mol_init[3][1][0:2], -a)

                        # Centering Molecule in Box:
                        ms, mm = mol_size(Mol)
                        offset = np.array([ (bs - s)/2. - m for s, m in zip(ms,mm)])
                        for k, off in enumerate(offset):
                            for j in range(len(Mol)):
                                Mol[j][1][k] += off
                        print(Mol)
                        
                        cell         = gto.Cell()
                        cell.a       = [[bs, 0., 0.], [0., bs, 0.], [0., 0., bs]] 
                        cell.unit    = 'B'
                        cell.verbose = 3
                        cell.basis   = basis
                        cell.pseudo  = 'gth-pade'
                        cell.mesh    = np.array(mesh)
                        cell.atom    = Mol
                        cell.build()
                       
                        f.write("Number of Built in basis functions: \n")
                        f.write(str(cell.nao_nr()) + "\n")

                        overlap = cell.pbc_intor('int1e_ovlp_sph')
                        w, _ = la.eig(overlap)
                        max_ev[i] = np.amax(w)
                        min_ev[i] = np.amin(w)
                        con_no[i] = np.amax(w)/ np.amin(w)
                        m[i] = len(w[w<10**-5])
                        
                        print("Max eigenvalue: "   , max_ev[i])
                        print("Min eigenvalue: "   , min_ev[i])
                        print("Condition no.: "    , con_no[i])
                        print("No. of EV<10e-5: "  , m[i])

                        # VDG calculations
                        print("Creating  VDG Hamiltonian ...")
                        log.write("        Creating  VDG Hamiltonian ...\n")
                        start_H = time.time()

                        log.write("            Creating Voronoi decomposition ... \n")
                        start = time.time()
                        
                        atoms_2d = np.array([atom[1][:2] for atom in cell.atom])
                        V_net = dg_tools.get_V_net_per(atoms_2d, 0, cell.a[0][0],0, cell.a[1][1])
                        voronoi_cells = dg_tools.get_V_cells(V_net, atoms_2d)
                        
                        log.write("            Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                        log.write("            VDG Hamiltonian ... \n")
                        start = time.time()
                        
                        cell_vdg = dg.dg_model_ham(cell, None ,'rel_num', accc, True, voronoi_cells)
                        f.write("Number of DG basis functions:\n")
                        f.write(str(cell_vdg.nao) + "\n")
                        log.write("            Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                        log.write("        Done! Elapsed time: " + str(time.time() - start_H) + "sec.\n")

                        overlap_dg = cell_vdg.ovl_dg
                        w_dg, _ = la.eig(overlap_dg)
                        max_ev_dg[i] = np.amax(w_dg)
                        min_ev_dg[i] = np.amin(w_dg)
                        con_no_dg[i] = np.amax(w_dg)/ np.amin(w_dg)
                        m_dg[i] = len(w_dg[w_dg<10**-5])
                        
                        print("Max eigenvalue (DG): "   , max_ev_dg[i])
                        print("Min eigenvalue (DG): "   , min_ev_dg[i])
                        print("Condition no. (DG): "    , con_no_dg[i])
                        print("No. of EV<10e-5 (DG): "  , m_dg[i])


                        # HF in VDG
                        log.write("        Computing HF in VDG Bases ...\n")
                        start = time.time()
                        
                        mfe_dg[i] = cell_vdg.run_RHF()
                        
                        log.write("        Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                    
                        # MP2 in VDG
                        log.write("        Computing MP2 in VDG Bases ...\n")
                        start = time.time()
                        
                        mpe_dg[i], _ = cell_vdg.run_MP2()
                        
                        log.write("        Done! Elapsed time: " + str(time.time() - start) + "sec.\n")

                        # CCSD in VDG
                        log.write("        Computing CCSD in VDG Bases ...\n")
                        start = time.time()
                        
                        cce_dg[i], _ = cell_vdg.run_CC()
                        
                        log.write("        Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                        
                        del cell_vdg
                        
                        # HF in builtin
                        log.write("        Comuting HF using PyScf-PBC...\n")
                        start = time.time()
                        
                        mf = scf.RHF(cell, exxdiv='ewald') # madelung correction
                        mf.kernel(dump_chk = False)
                        mfe[i] = mf.e_tot
                        
                        log.write("        Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                        
                        # MP2 in builtin
                        log.write("        Comuting MP2 using PyScf-PBC...\n")
                        start = time.time()
                        
                        mpe[i], _ = mp.MP2(mf).kernel()
                        
                        log.write("        Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                            
                        # CCSD in builtin
                        log.write("        Comuting CCSD using PyScf-PBC...\n")
                        start = time.time()
                        
                        cc_bi = cc.CCSD(mf)
                        cc_bi.kernel()
                        cce[i] =cc_bi.e_corr
                        log.write("        Done! Elapsed time: " + str(time.time() - start) + "sec.\n")
                        
                        del cell


                    f.write("Mean-field energy: " + str(mfe)    + "\n")
                    f.write("MP2 corr. energy : " + str(mpe)    + "\n")
                    f.write("CCSD corr. energy: " + str(cce)    + "\n")
                    f.write("Max eigenvalue   : " + str(max_ev) + "\n") 
                    f.write("Min eigenvalue   : " + str(min_ev) + "\n")
                    f.write("Condition no.    : " + str(con_no) + "\n")
                    f.write("No. of EV<10e-5  : " + str(m)      + "\n")
                    
                    f.write("Mean-field energy (DG): " + str(mfe_dg)    + "\n")
                    f.write("MP2 corr. energy (DG) : " + str(mpe_dg)    + "\n")
                    f.write("CCSD corr. energy (DG): " + str(cce_dg)    + "\n")
                    f.write("Max eigenvalue (DG)   : " + str(max_ev_dg) + "\n")
                    f.write("Min eigenvalue (DG)   : " + str(min_ev_dg) + "\n")
                    f.write("Condition no. (DG)    : " + str(con_no_dg) + "\n")
                    f.write("No. of EV<10e-5 (DG)  : " + str(m_dg)      + "\n")

            
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

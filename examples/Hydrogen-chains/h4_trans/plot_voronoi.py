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
        out   = np.round(r_vec + x,6)
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

        #test_trans()
        angles = np.linspace(0, np.pi/2.0, num=7)
        #angles = np.array([np.pi/2.])

        boxsizes = np.array([12])
        dgrid    = [5] * 3 
        #bases    = ['augccpvdz', '631++g', '321++g', 'ccpvdz', '631g', '321g']
        #acc      = [.99, .99, .99, .99, .99, .99]
        bases = ['321g']
        acc   = [.99]

        Mol_init = [['H', [0,0,0]],['H', [2,0,0]], ['H', [0,2,0]], ['H', [2,2,0]]]
    
        mfe = np.zeros(len(angles))
        max_ev = np.zeros(len(angles))
        min_ev = np.zeros(len(angles))
        con_no = np.zeros(len(angles))
        m      = np.zeros(len(angles)) 
        
        mfe_dg = np.zeros(len(angles))
        max_ev_dg = np.zeros(len(angles))
        min_ev_dg = np.zeros(len(angles))
        con_no_dg = np.zeros(len(angles))
        m_dg      = np.zeros(len(angles)) 

        for bs in boxsizes:
            
            # discretization mesh
            mesh = [int(d * x) for d, x in zip(dgrid, [bs]*3)]

            # Centering Molecule in Box:
            #ms = mol_size(Mol_init)
            #offset = np.array([ (bs - s)/2. for s in ms])
            #Mol_ic = copy.deepcopy(Mol_init)
            #for j, mol in enumerate(Mol_ic):
            #    for k, off in enumerate(offset):
            #        Mol_ic[j][1][k] += off
            #Mol = copy.deepcopy(Mol_ic)            
            
            for ac, basis in enumerate(bases):
                for i, a in enumerate(angles):
                   
                    Mol = copy.deepcopy(Mol_init)
                    Mol[2][1][0:2] = trans(Mol_init[0][1][0:2], Mol_init[2][1][0:2], a)
                    Mol[3][1][0:2] = trans(Mol_init[1][1][0:2], Mol_init[3][1][0:2], -a)
                    
                    print(Mol)

                    # Centering Molecule in Box:
                    ms, mm = mol_size(Mol)
                    offset = np.array([ (bs - s)/2. - m for s, m in zip(ms,mm)])
                    for k, off in enumerate(offset):
                        for j in range(len(Mol)):
                            Mol[j][1][k] += off
                    
                    cell         = gto.Cell()
                    cell.a       = [[bs, 0., 0.], [0., bs, 0.], [0., 0., bs]] 
                    cell.unit    = 'B'
                    cell.verbose = 3
                    cell.basis   = basis
                    cell.pseudo  = 'gth-pade'
                    cell.mesh    = np.array(mesh)
                    cell.atom    = Mol
                    cell.build()
                   
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
                    start_H = time.time()

                    start = time.time()
                    
                    atoms_2d = np.array([atom[1][:2] for atom in cell.atom])
                    V_net = dg_tools.get_V_net_per(atoms_2d, 0, cell.a[0][0],0, cell.a[1][1])
                    voronoi_cells = dg_tools.get_V_cells(V_net, atoms_2d)
                    
                    start = time.time()
                   
                    cell_vdg = dg.dg_model_ham(cell, None ,'rel_num', acc[ac], True, voronoi_cells, V_net)
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

                    del cell
                    del cell_vdg

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

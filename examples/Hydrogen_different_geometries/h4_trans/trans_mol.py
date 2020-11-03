from pyscf import gto, scf, cc, ao2mo, fci
from pyscf.fci import direct_spin0
import numpy as np
from numpy import linalg as la
import matplotlib
import matplotlib.pyplot as plt
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

def test_trans():
    x1 = np.array([1, 1])
    x2 = np.array([1, 0.5])
    p1 = np.array([2.75, 1]) 
    p2 = np.array([2.75, 0.5])

    angles = np.linspace(0, np.pi/2.0, num=12)
    for i, a in enumerate(angles):
        pn1 = trans(x1, p1, a)
        pn2 = trans(x2, p2, -a)

        plt.plot(pn1[1], pn2[0], 'r+')
        plt.plot(pn2[1], pn2[0], 'b+')
    plt.plot(x1[1], x1[0], 'ro')
    plt.plot(x2[1], x2[0], 'bo')
    plt.show()
        

if __name__ == '__main__':

    #test_trans()
    angles = np.linspace(0, np.pi/2.0, num=7)

    Mol_init = [['H', [0,0,0]],['H', [2,0,0]], ['H', [0,2,0]], ['H', [2,2,0]] ]
    Mol      = copy.deepcopy(Mol_init)
    
    mfe      = np.zeros(len(angles))
    gamma_hf = np.zeros(len(angles))
    ecc      = np.zeros(len(angles))
    efci     = np.zeros(len(angles))
    max_ev   = np.zeros(len(angles))
    min_ev   = np.zeros(len(angles))
    con_no   = np.zeros(len(angles))
    m        = np.zeros(len(angles)) 
    for i, a in enumerate(angles):
        
        Mol[2][1][0:2] = trans(Mol_init[0][1][0:2], Mol_init[2][1][0:2], a)
        Mol[3][1][0:2] = trans(Mol_init[1][1][0:2], Mol_init[3][1][0:2], -a)
        mol = gto.M()
        mol.basis = 'aug-ccpvdz'
        mol.atom  = Mol
        mol.unit  = 'bohr'
        mol.verbose = 3
        mol.build()
        
        overlap = mol.intor('int1e_ovlp_sph')
        #
        w, _ = la.eig(overlap)
        max_ev[i] = np.amax(w)
        min_ev[i] = np.amin(w)
        con_no[i] = np.amax(w)/ np.amin(w)
        m[i] = len(w[w<10**-5])
        
        
        print("Maximial EV: ",  max_ev[i])
        print("Minimal EV: ", min_ev[i])
        print("Cond. no.: ", con_no[i])
        print("m (10e-5): ", m[i])


        mf     = mol.RHF()
        mf.kernel()
        mfe[i] = mf.e_tot
    
        e_homo, e_lumo = (
            mf.mo_energy[mol.nelectron // 2 - 1],
            mf.mo_energy[mol.nelectron // 2],
        )
        gamma_hf[i] = e_lumo - e_homo
        print("HOMO-LUMO:", gamma_hf[i])
        
        mycc      = cc.CCSD(mf)
        mycc.kernel()
        ecc[i]    = mycc.e_tot

        #h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
        #eri = ao2mo.kernel(mol, mf.mo_coeff)
        #cisolver = fci.direct_spin1.FCI(mol)
        #efci[i], ci = cisolver.kernel(h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc())


    plt.plot(angles*180/np.pi, mfe, 'b-v', label =  'HF  (' + mol.basis + ')')
    plt.plot(angles*180/np.pi, ecc, 'r-x', label =  'CC  (' + mol.basis + ')')
    #plt.plot(angles*180/np.pi, efci, 'r-o', label =  'FCI  (' + mol.basis + ')')
    plt.title('PES in ' + mol.basis)
    plt.legend()
    plt.show()

    plt.plot(angles*180/np.pi,gamma_hf, label="HOMO-LUMO")
    plt.title('HOMO-LUMO gap ' + mol.basis)
    plt.legend()
    plt.show()

    plt.plot(angles*180/np.pi,max_ev, label="Maximal EV")
    plt.title('Maximial EV ' + mol.basis)
    plt.legend()
    plt.show()

    plt.plot(angles*180/np.pi,min_ev, label="Minimal ev")
    plt.title('Minimal EV ' + mol.basis)
    plt.legend()
    plt.show()

    plt.plot(angles*180/np.pi,con_no, label="Cond.-no")
    plt.title('Condition number ' + mol.basis)
    plt.legend()
    plt.show()
    
    plt.plot(angles*180/np.pi,m, label="m")
    plt.title('Number of EV < 10e-5 in ' + mol.basis)
    plt.legend()
    plt.show()

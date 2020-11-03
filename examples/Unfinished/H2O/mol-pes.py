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
from pyscf import gto, scf

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
    Mol  = [['H',[0,-1.684338648, 0]], ['O',[0,0,0]], ['H',[0,-1.684338648,0 ]]]
    Mol1 = copy.deepcopy(Mol)

    #angles = np.linspace(0, np.pi, num=20)
    #angles = angles[0:16]
    # 104.476/2 *pi/180 = 0.911725...
    #
    angles = np.array([0.9117250946567979, np.pi/2.0])

    mfe_dzp = np.zeros(len(angles))
    mfe_tzp = np.zeros(len(angles))
    mfe_qzp = np.zeros(len(angles))

    for k, angle in enumerate(angles):
        Mol[0][1][0:2] = rot(angle, Mol1[0][1][0:2])
        Mol[2][1][0:2] = rot(-angle, Mol1[2][1][0:2])

        # Placing Molecule in the box (Oxigen in the center!) :

        offset    = np.array([10,10,3])
        atom_pos  = np.array([atom[1] for atom in Mol])
        atoms     = [atom[0] for atom in Mol]
        atom_off  = [3,3,1.5]
        atoms_box = np.array([pos + atom_off for pos in atom_pos])
        Mol_box   = [[atoms[i],atoms_box[i]] for i in range(len(atoms))]
        Mol_size = np.array([0,0,0])
        X = np.array([int(np.ceil( off + bl)) for off, bl in zip(offset, Mol_size)])

        mol_dzp = gto.M()
        mol_tzp = gto.M()
        mol_qzp = gto.M()

        mol_dzp.atom = Mol_box
        mol_tzp.atom = Mol_box
        mol_qzp.atom = Mol_box
        
        mol_dzp.unit    = 'B'
        mol_tzp.unit    = 'B'
        mol_qzp.unit    = 'B'

        mol_dzp.verbose = 3
        mol_tzp.verbose = 3
        mol_qzp.verbose = 3

        mol_dzp.basis   = 'dzp'
        mol_tzp.basis   = 'tzp'
        mol_qzp.basis   = 'qzp'
        
        mol_dzp.build()
        mol_tzp.build()
        mol_qzp.build()
        

        myhf_dzp = scf.RHF(mol_dzp)
        myhf_tzp = scf.RHF(mol_tzp)
        myhf_qzp = scf.RHF(mol_qzp)

        mfe_dzp[k] = myhf_dzp.kernel()
        mfe_tzp[k] = myhf_tzp.kernel()
        mfe_qzp[k] = myhf_qzp.kernel()
        

    print("Mean field in dzp:")
    print("  Minimum:", np.amin(mfe_dzp))
    print("  Maximum:", np.amax(mfe_dzp))
    print("  abs diff:", np.abs(np.amin(mfe_dzp) - np.amax(mfe_dzp)))
    print("Mean field in tzp:")
    print("  Maximum:", np.amin(mfe_tzp))
    print("  Minimum:", np.amax(mfe_tzp))
    print("  Abs diff:", np.abs(np.amin(mfe_tzp) - np.amax(mfe_tzp)))
    print("Mean field in qzp:")
    print("  Maximum:", np.amin(mfe_qzp))
    print("  Minimum", np.amax(mfe_qzp))
    print("  Abs diff:", np.abs(np.amin(mfe_qzp) - np.amax(mfe_qzp)))

    # old angle offset: 33.129
    plt.plot(2* angles*180/np.pi , mfe_dzp, 'b-v', label =  'HF  (dzp)')
    plt.plot(2* angles*180/np.pi , mfe_tzp, 'r-v', label =  'HF  (tzp)')
    plt.plot(2* angles*180/np.pi , mfe_qzp, 'g-v', label =  'HF  (qzp)')
    plt.legend()
    plt.show()


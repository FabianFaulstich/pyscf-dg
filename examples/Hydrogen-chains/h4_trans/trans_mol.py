from pyscf import gto, scf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def trans(x, p, a):
    '''
    a-rotation of p around x
    '''
    c, s  = np.cos(a), np.sin(a)
    R     = np.array(((c, -s), (s, c)))
    vec   = p - x
    r_vec = np.dot(R,vec)
    return r_vec + x

def test_trans():
    x1 = np.array([1, 1])
    x2 = np.array([1, 0.5])
    p1 = np.array([2.75, 1]) 
    p2 = np.array([2.75, 0.5])

    angles = np.linspace(0, np.pi, num=12)
    for i, a in enumerate(angles):
        p1 = trans(x1, p1, a)
        p2 = trans(x2, p2, -a)

        plt.plot(p1[0], p2[1], 'r+')
        plt.plot(p2[0], p2[1], 'b+')
    plt.plot(x1[0], x1[1], 'ro')
    plt.plot(x2[0], x2[1], 'bo')
    plt.show()
        

if __name__ == '__main__':

    test_trans()

    exit()
    mol = gto.M()
    mol.basis = 'ccpvdz'
    mol.atom  = Mol
    mol.unit  = 'bohr'
    mol.verbose = 3
    mol.build()

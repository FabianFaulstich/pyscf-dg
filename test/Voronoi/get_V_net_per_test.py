import sys
sys.path.append('../../src')
import dg_tools as dg

import numpy as np
import matplotlib.pyplot as plt

# Testing cubical symmetry
def test_cube(detail=False):
    atoms = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
    x_max =  3
    x_min = -3
    y_max =  3
    y_min = -3

    V_net = dg.get_V_net_per(atoms, x_min, x_max, y_min, y_max)
    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')
    for v in V_net:
        for c in v[1]:
            plt.plot([v[0][0],V_net[c][0][0]],[v[0][1],V_net[c][0][1]],'r-')
    plt.xlim(x_min -1.5,x_max +1.5)
    plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

    if detail:
        for v in V_net:
            # plotting atoms
            plt.plot(atoms[:,0], atoms[:,1], 'bo')

            # plotting all vertices
            plt.plot(vert[:,0], vert[:,1],'ro')

            # plotting considered vertex
            plt.plot(v[0][0],v[0][1],'kD')

            # plotting connected vertices
            for c in v[1]:
                plt.plot(V_net[c][0][0], V_net[c][0][1] ,'k*')
                plt.plot([v[0][0],V_net[c][0][0]],[v[0][1],V_net[c][0][1]],'k-')
            plt.xlim(x_min -1.5,x_max +1.5)
            plt.ylim(x_min -1.5,x_max +1.5)
            plt.show()

def test_chain_H4(detail = False):

    atoms = np.array([[-3, 0], [-1, 0],
                      [1, 0], [3, 0]])

    x_max =  6
    x_min = -6
    y_max =  6
    y_min = -6

    V_net = dg.get_V_net_per(atoms, x_min, x_max, y_min, y_max)
    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')
    for v in V_net:
        for c in v[1]:
            plt.plot([v[0][0],V_net[c][0][0]],[v[0][1],V_net[c][0][1]],'r-')
    plt.xlim(x_min -1.5,x_max +1.5)
    plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

    if detail:
        for v in V_net:
            # plotting atoms
            plt.plot(atoms[:,0], atoms[:,1], 'bo')

            # plotting all vertices
            plt.plot(vert[:,0], vert[:,1],'ro')

            # plotting considered vertex
            plt.plot(v[0][0],v[0][1],'kD')

            # plotting connected vertices
            for c in v[1]:
                plt.plot(V_net[c][0][0], V_net[c][0][1] ,'k*')
                plt.plot([v[0][0],V_net[c][0][0]],[v[0][1],V_net[c][0][1]],'k-')
            plt.xlim(x_min -1.5,x_max +1.5)
            plt.ylim(x_min -1.5,x_max +1.5)
            plt.show()

def test_d_chain_H4(detail = False):

    atoms = np.array([[-3, .3], [-1, -.1],
                      [1, .2], [3, -.4]])

    x_max =  6
    x_min = -6
    y_max =  6
    y_min = -6

    V_net = dg.get_V_net_per(atoms, x_min, x_max, y_min, y_max)
    
    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')
    for v in V_net:
        for c in v[1]:
            plt.plot([v[0][0],V_net[c][0][0]],[v[0][1],V_net[c][0][1]],'r-')
    plt.xlim(x_min -1.5,x_max +1.5)
    plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

    if detail:
        for v in V_net:
            # plotting atoms
            plt.plot(atoms[:,0], atoms[:,1], 'bo')

            # plotting all vertices
            plt.plot(vert[:,0], vert[:,1],'ro')

            # plotting considered vertex
            plt.plot(v[0][0],v[0][1],'kD')

            # plotting connected vertices
            for c in v[1]:
                plt.plot(V_net[c][0][0], V_net[c][0][1] ,'k*')
                plt.plot([v[0][0],V_net[c][0][0]],[v[0][1],V_net[c][0][1]],'k-')
            plt.xlim(x_min -1.5,x_max +1.5)
            plt.ylim(x_min -1.5,x_max +1.5)
            plt.show()


if __name__ == '__main__':

    # Testing cubical symmetry 
    test_cube()

    # Testing quasi 1D systems
    test_chain_H4()

    # Testing distorted quasi 1D systems
    test_d_chain_H4()

    # Testing Ethylene symmetry
    test_C2H4()



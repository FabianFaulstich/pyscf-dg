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

    V_net = dg.get_V_net(atoms, x_min, x_max, y_min, y_max)
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

# Testing distorted cubical symmetry
def test_d_cube(detail=False):
    atoms = np.array([[1.5,.5],[1.25,-.5],[-1,0.8],[-1,-1]])
    x_max =  3
    x_min = -3
    y_max =  3
    y_min = -3

    V_net = dg.get_V_net(atoms, x_min, x_max, y_min, y_max)
    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')
    for v in V_net:
        for c in v[1]:
            plt.plot(V_net[c][0][0], V_net[c][0][1] ,'k*')
            plt.plot([v[0][0],V_net[c][0][0]],[v[0][1],V_net[c][0][1]],'k-')
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

# Testing Ethylene structure
def test_C2H4(detail=False):
    atoms = np.array([[-2.4, -1.86], [-2.4, 1.86], [-1.34, 0], [1.34, 0],[2.4,-1.86], [2.4,1.86]])

    x_max =  6
    x_min = -6
    y_max =  6
    y_min = -6

    V_net = dg.get_V_net(atoms, x_min, x_max, y_min, y_max)
    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')
    for v in V_net:
        for c in v[1]:
            plt.plot(V_net[c][0][0], V_net[c][0][1] ,'k*')
            plt.plot([v[0][0],V_net[c][0][0]],[v[0][1],V_net[c][0][1]],'k-')
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
    test_cube(detail=True)
    
    # Testing distorted cubical symmetry
    test_d_cube()

    # Testing Ethylene symmetry
    test_C2H4()

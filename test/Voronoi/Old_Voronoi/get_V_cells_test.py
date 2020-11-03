import sys
sys.path.append('../../src')
import dg_tools as dg

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


# Testing cubical symmetry
def test_cube(detail = False):
    atoms = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
    x_max =  3
    x_min = -3
    y_max =  3
    y_min = -3

    V_net   = dg.get_V_net_per(atoms, x_min, x_max, y_min, y_max)
    V_cells = dg.get_V_cells(V_net, atoms)    
    
    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')
    
    for vcell in V_cells:
        for i in range(len(vcell)-1):
            plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
    plt.xlim(x_min -1.5,x_max +1.5)
    plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

    if detail == True:
        for vcell in V_cells:
            plt.plot(atoms[:,0], atoms[:,1], 'bo')
            plt.plot(vert[:,0], vert[:,1],'ro')
            for i in range(len(vcell)-1):
                plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
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
    
    V_net   = dg.get_V_net_per(atoms, x_min, x_max, y_min, y_max)
    V_cells = dg.get_V_cells(V_net, atoms)

    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')

    for vcell in V_cells:
        for i in range(len(vcell)-1):
            plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
    plt.xlim(x_min -1.5,x_max +1.5)
    plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

    if detail == True:
        for vcell in V_cells:
            plt.plot(atoms[:,0], atoms[:,1], 'bo')
            plt.plot(vert[:,0], vert[:,1],'ro')
            for i in range(len(vcell)-1):
                plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
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

    V_net   = dg.get_V_net_per(atoms, x_min, x_max, y_min, y_max)
    V_cells = dg.get_V_cells(V_net, atoms)

    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')

    for vcell in V_cells:
        for i in range(len(vcell)-1):
            plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
    plt.xlim(x_min -1.5,x_max +1.5)
    plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

    if detail == True:
        for vcell in V_cells:
            plt.plot(atoms[:,0], atoms[:,1], 'bo')
            plt.plot(vert[:,0], vert[:,1],'ro')
            for i in range(len(vcell)-1):
                plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
            plt.xlim(x_min -1.5,x_max +1.5)
            plt.ylim(x_min -1.5,x_max +1.5)
            plt.show()

# Testing periodic quasi 1D systems

def test_chain_H4(detail = False):

    atoms = np.array([[-3, 0], [-1, 0],
                      [1, 0], [3, 0]])

    x_max =  6
    x_min = -6
    y_max =  6
    y_min = -6

    V_net   = dg.get_V_net_per(atoms, x_min, x_max, y_min, y_max)
    V_cells = dg.get_V_cells(V_net, atoms)

    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')

    for vcell in V_cells:
        for i in range(len(vcell)-1):
            plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
    plt.xlim(x_min -1.5,x_max +1.5)
    plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

    if detail == True:
        for vcell in V_cells:
            plt.plot(atoms[:,0], atoms[:,1], 'bo')
            plt.plot(vert[:,0], vert[:,1],'ro')
            for i in range(len(vcell)-1):
                plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
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
    V_cells = dg.get_V_cells(V_net, atoms)

    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')

    for vcell in V_cells:
        for i in range(len(vcell)-1):
            plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
    plt.xlim(x_min -1.5,x_max +1.5)
    plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

    if detail == True:
        for vcell in V_cells:
            plt.plot(atoms[:,0], atoms[:,1], 'bo')
            plt.plot(vert[:,0], vert[:,1],'ro')
            for i in range(len(vcell)-1):
                plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
            plt.xlim(x_min -1.5,x_max +1.5)
            plt.ylim(x_min -1.5,x_max +1.5)
            plt.show()

def test_chain_H2(detail = False):

    atoms = np.array([[5.4, 3], [6.6, 3]])

    x_max = 12
    x_min = 0
    y_max = 6 
    y_min = 0

    V_net = dg.get_V_net_per(atoms, x_min, x_max, y_min, y_max)
    V_cells = dg.get_V_cells(V_net, atoms)

    vert = np.array([elem[0] for elem in V_net])

    plt.plot(atoms[:,0], atoms[:,1], 'bo')
    plt.plot(vert[:,0], vert[:,1],'ro')

    for vcell in V_cells:
        for i in range(len(vcell)-1):
            plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
    plt.xlim(x_min -1.5,x_max +1.5)
    plt.ylim(x_min -1.5,x_max +1.5)
    plt.show()

    if detail == True:
        for vcell in V_cells:
            plt.plot(atoms[:,0], atoms[:,1], 'bo')
            plt.plot(vert[:,0], vert[:,1],'ro')
            for i in range(len(vcell)-1):
                plt.plot([vcell[i][0],vcell[i+1][0]],[vcell[i][1],vcell[i+1][1]], 'k-')
            plt.xlim(x_min -1.5,x_max +1.5)
            plt.ylim(x_min -1.5,x_max +1.5)
            plt.show()


if __name__ == '__main__':

    # Testing H2
    test_chain_H2(True)

    # Testing periodic quasi 1D system
    #test_chain_H4(True)

    # Testing periodic distorted 1D system
    test_d_chain_H4(True)

    # Testing cubical symmetry 
    test_cube()

    # Testing distorted cubical symmetry
    test_d_cube()

    # Testing Ethylene symmetry
    test_C2H4(True)

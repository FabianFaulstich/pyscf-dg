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

from scipy.spatial import Voronoi,ConvexHull

import matplotlib.pyplot as plt

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
        out.append(expan([mol[i] for mol in Mol]))
        m.append(minim([mol[i] for mol in Mol]))
    return np.array(out), m

if __name__ == '__main__':

    #test_trans()
    angles = np.linspace(0, np.pi/2.0, num=7)

    for angle in angles:
        
        fig,  ax  = plt.subplots(nrows=1, ncols=1)
        fig1, ax1 = plt.subplots(nrows=1, ncols=1)

        x_max = 12
        x_min = 0
        y_max = 12
        y_min = 0
        bs = 12 
        
        Mol_init = [[0,0,0], [2,0,0], [0,2,0], [2,2,0]]

                   
        Mol = copy.deepcopy(Mol_init)
        Mol[2][0:2] = trans(Mol_init[0][0:2], Mol_init[2][0:2], angle)
        Mol[3][0:2] = trans(Mol_init[1][0:2], Mol_init[3][0:2], -angle)
        
        print(Mol)

        # Centering Molecule in Box:
        ms, mm = mol_size(Mol)
        offset = np.array([ (bs - s)/2. - m for s, m in zip(ms,mm)])
        for k, off in enumerate(offset):
            for j in range(len(Mol)):
                Mol[j][k] += off

        atoms = np.array([mol[:2] for mol in Mol])

        
        atoms_tile = dg_tools.tile_atoms(atoms,x_max - x_min, y_max - y_min)
        voronoi  = Voronoi(atoms_tile)

        # plotting conections between two voronoi vertices: 
        for rv in voronoi.ridge_vertices:
            rv = np.asarray(rv)
            if np.all(rv >= 0):
                ax1.plot(voronoi.vertices[rv,0], voronoi.vertices[rv,1], 'b-')

        # plotting connextions from box limit with voronoi
        center = atoms_tile.mean(axis=0)
        ax1.plot(center[0],center[1],'r+')
        for pointidx, rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
            rv = np.asarray(rv)
            if np.any(rv < 0):
                i = rv[rv >= 0][0]
                t = atoms_tile[pointidx[1]]- atoms_tile[pointidx[0]]
                t = t/np.linalg.norm(t)
                n = np.array([-t[1],t[0]])
                midpoint = atoms_tile[pointidx].mean(axis=0)
                far_point = voronoi.vertices[i] + np.sign(np.dot(midpoint-center, n)) * n * 100
                ax1.plot([voronoi.vertices[i,0],far_point[0]], [voronoi.vertices[i,1], far_point[1]], 'b-')

        # plotting voronoi vertices:
        ax1.plot(voronoi.vertices[:,0], voronoi.vertices[:,1],'kD', markersize = 3)  # poltting voronoi vertices
        ax1.plot(atoms_tile[:,0], atoms_tile[:,1], 'ko')
        ax1.plot([[x_min,x_min],[x_min,x_max],[x_max,x_max],[x_max,x_min]],[[y_min,y_max],[y_max,y_max],[y_max,y_min],[y_min,y_min]],'r-')
        ax1.set_xlim(x_min-(x_max-x_min),x_max+(x_max-x_min))
        ax1.set_ylim(y_min-(y_max-y_min),y_max+(y_max-y_min))



        V_net = dg_tools.get_V_net_per(atoms, x_min, x_max, y_min, y_max)
        V_cells = dg_tools.get_V_cells(V_net, atoms)
        vert = np.array([elem[0] for elem in V_net])
        V_cells_l = V_cells.tolist()
    
        if angle > 0 and angle < np.pi/2.0:
            for v in V_net:
                if np.round(v[0][0],6) == 0 and np.round(v[0][1],6) == 0:
                    appnd = dg_tools.get_cell(v[0] + np.array([0.1, 0.1]),V_net).tolist()
                    V_cells_l.append(appnd) 

                elif np.round(v[0][0],6) == 12 and np.round(v[0][1],6) == 0:
                    appnd = dg_tools.get_cell(v[0] + np.array([-0.1, 0.1]),V_net).tolist()
                    V_cells_l.append(appnd)

                elif np.round(v[0][0],6) == 6 and np.round(v[0][1],6) == 12:
                    appnd = [dg_tools.get_cell(v[0] + np.array([-.1 ,-0.05]),V_net).tolist()]
                    appnd.append(dg_tools.get_cell(v[0] + np.array([.1 ,-0.05]),V_net).tolist())
                    V_cells_l = V_cells_l + appnd
       
        #coords = np.mgrid[0:12:0.1, 0:12:0.1].reshape(2,-1).T
        
        #idx_mat = []
        #vstart = time.time()
        #for vcell in V_cells:
        #    idx_mat.append(dg_tools.in_hull(coords, vcell))
        #idx_mat = np.array(idx_mat).transpose()

        # Asigning boundary values to a cell
        #for i, elem in enumerate(idx_mat):
        #    if len(elem[elem == True]) == 0:
        #        k = get_dist_atom(atoms, 12, 12, coords[i])
        #        idx_mat[i,k] = True
        #    elif len(elem[elem == True]) > 1:
        #        elem_n = np.zeros_like(elem, dtype = bool)
        #        elem_n[np.where(elem == True)[0][0]] = True
        #        idx_mat[i,:] = elem_n
        #vend = time.time()
        #print("Done! Elapsed time for mixed voronoi: ", vend - vstart)
        

        cmap = dg_tools.get_cmap(5)
        #for i, col in enumerate(idx_mat.transpose()):
        #    for k, point in enumerate(col):
        #        if point:
        #            ax.plot(coords[k][0],coords[k][1], color =cmap(i) , marker='s', markersize = 10)
      
        ax.fill(np.array(V_cells_l[0])[:,0],np.array(V_cells_l[0])[:,1], color = cmap(0))
        ax.fill(np.array(V_cells_l[1])[:,0],np.array(V_cells_l[1])[:,1], color = cmap(1))
        ax.fill(np.array(V_cells_l[2])[:,0],np.array(V_cells_l[2])[:,1], color = cmap(2))
        ax.fill(np.array(V_cells_l[3])[:,0],np.array(V_cells_l[3])[:,1], color = cmap(3))
        if len(V_cells_l) > 4:
            for V_cell in V_cells_l:
                V_cell = np.array(V_cell)
                for v in V_cell:
                    if np.round(v[0],6) == 0 and np.round(v[1],6) == 0:
                        ax.fill(V_cell[:,0],V_cell[:,1], color = cmap(2))
                    elif np.round(v[0],6) == 12 and np.round(v[1],6) == 0:
                        ax.fill(V_cell[:,0],V_cell[:,1], color = cmap(3))
                    elif np.round(v[0],6) == 6 and np.round(v[1],6) == 12:
                        for w in V_cell:
                            if np.round(w[0],6) < 6-0.5:
                                print(V_cell)
                                ax.fill(V_cell[:,0],V_cell[:,1], color = cmap(0))
                            if np.round(w[0],6) > 6+0.5:
                                print(V_cell)
                                ax.fill(V_cell[:,0],V_cell[:,1], color = cmap(1))


        #plt.show()
        #exit()
        #for vcell in V_cells_l:
        #    vcell = np.array(vcell)
        #    ax.fill(vcell[:,0],vcell[:,1])

        ax.plot(atoms[:,0], atoms[:,1], 'ko', markersize=10)
        ax.plot(vert[:,0], vert[:,1],'kD', markersize = 3)
        for v in V_net:
            for c in v[1]:
                ax.plot([v[0][0],V_net[c][0][0]],[v[0][1],V_net[c][0][1]],'b-')
        ax.plot(vert[:,0], vert[:,1],'kD', markersize = 5)
        
        #ax.xlim(x_min -1.5,x_max +1.5)
        #ax.ylim(x_min -1.5,x_max +1.5)

        #ax.set_facecolor('dimgrey')    
        ax.axis('off')
        plt.show()
    





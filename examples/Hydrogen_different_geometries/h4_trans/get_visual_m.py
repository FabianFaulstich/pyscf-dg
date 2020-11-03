import sys
sys.path.append('../../../src')

import dg_model_ham as dg
import dg_tools
import numpy as np
from numpy import linalg as la
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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

from scipy.spatial import ConvexHull
from mayavi import mlab
import pyface.qt

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

        log = open("log_visual.txt", "w")
        f   = open("out_visual.txt", "w")
        angle = np.pi/4 
        
        boxsize  = 12
        dgrid    = [5] * 3 
        basis    = 'ccpvdz'
        acc      = .99 

        Mol = [['H', [0,0,0]],['H', [2,0,0]], ['H', [0,2,0]], ['H', [2,2,0]]]
        ms = mol_size(Mol)
            
        # discretization mesh
        mesh = [int(d * x) for d, x in zip(dgrid, [boxsize]*3)]

               
        Mol[2][1][0:2] = trans(Mol[0][1][0:2], Mol[2][1][0:2],  angle)
        Mol[3][1][0:2] = trans(Mol[1][1][0:2], Mol[3][1][0:2], -angle)

        # Centering Molecule in Box:
        ms, mm = mol_size(Mol)
        offset = np.array([ (boxsize - s)/2. - m for s, m in zip(ms,mm)])
        for k, off in enumerate(offset):
            for j in range(len(Mol)):
                Mol[j][1][k] += off
        print(Mol)
        
        cell         = gto.Cell()
        cell.a       = [[boxsize, 0., 0.], 
                        [0., boxsize, 0.], 
                        [0., 0., boxsize]] 
        cell.unit    = 'B'
        cell.verbose = 3
        cell.basis   = basis
        cell.pseudo  = 'gth-pade'
        cell.mesh    = np.array(mesh)
        cell.atom    = Mol
        cell.build()
       
        # VDG calculations
        print("Computing Gram Matrix ...")
        start = time.time()
        dg_gramm, dg_idx = dg.get_dg_gramm(cell, dg_cuts = None,
                    dg_trunc = 'rel_num', svd_tol = acc, voronoi = True,
                    dg_on = True, gram = None)
        print("Done! Elapsed time: ", time.time() - start)
       
        print(dg_gramm.shape)

        # Molecular geometry is in X-Y plane, extracting Z = 6.0 slice
        coords = cell.get_uniform_grids()
        idx_xy = np.where(coords[:,2] == 6)[0]
        dg_gramm = dg_gramm[idx_xy,:]
        orb  = dg_gramm[:,3]
        orb1 = dg_gramm[:,44] 
        orb  = orb.reshape((60,60))
        orb1 = orb1.reshape((60,60))
        
        coords = coords[coords[:,2] == 6]
        idx_mat = dg_tools.naive_voronoi(coords, 
                                         [mol[1] for mol in cell.atom], 
                                         12, 12, 12)
        fig = mlab.figure()

        x = np.unique(coords[:,0])
        y = np.unique(coords[:,1])

        X, Y = np.meshgrid(x, y)

        ax_ranges = [0, 12, 0, 12, 0, 12]
        ax_scale = [1.0, 1.0, 1.0]
        ax_extent = ax_ranges * np.repeat(ax_scale, 2)

        surf1 = mlab.surf(X, Y, orb, colormap='Oranges')
        #surf1.actor.actor.scale = ax_scale
        
        #mlab.view(60, 74, 17, [-2.5, -4.6, -0.3])
        #mlab.outline(surf1, color=(.7, .7, .7), extent=ax_extent)
        #mlab.axes(surf1, color=(.7, .7, .7), extent=ax_extent,
        #        ranges=ax_ranges,
        #        xlabel='x', ylabel='y', zlabel='z')
        mlab.show()
        exit()
        ax.plot_surface(X, Y, orb.T , 
                #rstride=1, cstride=1)
                cmap=cm.Oranges, linewidth=0, antialiased=False)


        
       
        vcell = idx_mat[:,0]
        points = coords[vcell,:]
        
        hullpoints = ConvexHull(points[:,:2])
        for simplex in hullpoints.simplices:
            ax.plot(points[simplex, 0], 
                    points[simplex, 1], 
                    points[simplex, 2]*0,
                    'r-', linewidth = 2)        
        


        #vcell = idx_mat[:,1]
        #points = coords[vcell,:]
        
        #hullpoints = ConvexHull(points[:,:2])
        #for simplex in hullpoints.simplices:
        #    ax.plot(points[simplex, 0], 
        #            points[simplex, 1], 
        #            points[simplex, 2]*0,
        #            'r-', linewidth = 2)        
        
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show() 
        exit()


        coords = coords[coords[:,2] == 6]
        print(coords[:5][:])
        coords = coords[:,:2]
        print(coords[:5][:])

        print(idx_xy.shape)
        print(coords.shape)
        print(dg_gramm[:,2].shape)

        log.close()
        f.close()

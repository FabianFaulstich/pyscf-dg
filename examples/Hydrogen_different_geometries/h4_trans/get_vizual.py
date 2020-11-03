import sys
sys.path.append('../../../src')

import dg_model_ham as dg
import dg_tools
import numpy as np
from numpy import linalg as la
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib import ticker
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
        orb  = dg_gramm[:,0]
        orb1 = dg_gramm[:,40]
        orb2 = dg_gramm[:,61]
        
        #coords = coords[coords[:,2] == 6]
        idx_mat = dg_tools.naive_voronoi(coords, 
                                         [mol[1] for mol in cell.atom], 
                                         12, 12, 12)
        #coords = coords[coords[:,2] == 6]
        x = np.unique(coords[:,0])
        y = np.unique(coords[:,1])

       
        dg_basis =  orb2.reshape((60,60)).T +\
                orb1.reshape((60,60)).T +\
                orb.reshape((60,60)).T 
       
        print(np.amin(dg_basis))
        print(np.amax(dg_basis))
        #dg_basis = np.abs(dg_basis)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=(14./2.54, 7./2.54))
        
        plt.contourf(x, y, dg_basis, 100, cmap = "seismic")

        cb = plt.colorbar(format='%.1f')
        tick_locator = ticker.MaxNLocator(nbins=6)
        cb.locator = tick_locator
        cb.update_ticks()
        cb.outline.set_visible(False)

        atoms = [mol[1][:2] for mol in cell.atom]
        for atom in atoms:
            plt.plot(atom[0],atom[1],'k.')

        plt.axis('off')
        ax = plt.gca()
        ax.text(.5, -.1, 'X-Axis', transform=ax.transAxes)
        ax.text(-.05, .5 , 'Y-Axis', rotation='vertical', transform=ax.transAxes)
       
        plt.plot(
                [6.15, 6.15, 11.79, 11.79, 7.54, 4.65, 0.00, 0.00, 6.15, 6.15, 
                    7.54, 6.15 - 1.39, 6.15, 6.15],
                [0.00, 7.60,  1.85,  0.61, 0.00, 0.00, 0.61, 1.85, 7.60, 11.52,
                    11.8, 11.8, 11.52, 11.8],
                ls = '-',
                c = 'silver',
                linewidth = 1.2
                )

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

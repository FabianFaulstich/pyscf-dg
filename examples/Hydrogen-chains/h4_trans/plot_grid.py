import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

import numpy as np
from scipy.optimize import curve_fit

def func(x,Y,a):
    return Y + a * x**(-3)

def get_cmp(n, name = 'hsv'):
    return plt.cm.get_cmap(name, n)

if __name__ == '__main__':

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    
    plt_b = False

    f   = open("out_grid_conv_leqOne.txt", "r")

    fig, arr1 = plt.subplots(nrows=2, ncols=4,  figsize=(10,8))

    for cnt, line in enumerate(f):
 
        if 'AO basis' in line:
            BS = line[9:]

        if 'Grid spacing' in line:
            fl    = [x for x in line.split()]
            h     = fl[3:]
            h[0]  = h[0][1:]
            grid  = [float(x[:-1]) for x in reversed(h)]
        
        if 'Mean-field energy:' in line:        
            fl    = [x for x in line.split()]
            h     = fl[2:]
            h[0]  = h[0][1:]
            h[-1] = h[-1][:-1]
            mf    = [float(x) for x in reversed(h)]
            
        if 'Mean-field energy (DG):' in line:
            fl    = [x for x in line.split()]
            h     = fl[3:]
            h[0]  = h[0][1:]
            h[-1] = h[-1][:-1]
            if h[-1] == '':
                h = h[:-1]
            mf_dg = [float(x) for x in reversed(h)]
        
        if 'MP2 corr. energy :' in line:
            fl    = [x for x in line.split()]
            h     = fl[4:]
            h[0]  = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mpe   = [float(x) for x in reversed(h)]

        if 'MP2 corr. energy (DG) :' in line:
            fl     = [x for x in line.split()]
            h      = fl[5:]
            h[0]   = h[0][1:]
            h[-1]  = h[-1][:-1]
            mpe_dg = [float(x) for x in reversed(h)]
            plt_b  = True


        if plt_b:
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    arr1[0,0].plot(grid, mf, '+-k' ,label = 'HF (aug)')
                    arr1[0,0].plot(grid, mf_dg, '+-r' ,label = 'HF (aug-DG)')
                    arr1[1,0].plot(grid, mpe, '+-k' ,label = 'MP2 (aug)')
                    arr1[1,0].plot(grid, mpe_dg, '+-r' ,label = 'MP2 (aug-DG)')
                    arr1[0,0].set_title("cc-pVDZ ")# + NDG + "/" + NBI )
                    arr1[0,0].set_xlim(grid[0], grid[-1])
                    arr1[1,0].set_xlim(grid[0], grid[-1])
                    arr1[0,0].legend()
                    arr1[1,0].legend()
                elif len(BS) < 8:
                    arr1[0,0].plot(grid, mf, 'x-k' ,label = 'HF')
                    arr1[0,0].plot(grid, mf_dg, 'x-r' ,label = 'HF (DG)')
                    arr1[1,0].plot(grid, mpe, 'x-k' ,label = 'MP2')
                    arr1[1,0].plot(grid, mpe_dg, 'x-r' ,label = 'MP2 (DG)')
                    arr1[1,0].set_title("cc-pVDZ ")#+ NDG + "/" + NBI)
                    arr1[0,0].set_xlim(grid[0], grid[-1])
                    arr1[1,0].set_xlim(grid[0], grid[-1])
                    arr1[0,0].legend()
                    arr1[1,0].legend()
            if '6311' in BS:
                if len(BS) > 7:
                    arr1[0,1].plot(grid, mf, '+-k' ,label = 'HF (aug)')
                    arr1[0,1].plot(grid, mf_dg, '+-r' ,label = 'HF (aug-DG)')
                    arr1[1,1].plot(grid, mpe, '+-k' ,label = 'MP2 (aug)')
                    arr1[1,1].plot(grid, mpe_dg, '+-r' ,label = 'MP2 (aug-DG)')
                    arr1[0,1].set_title("6-311G ")# + NDG + "/" + NBI )
                    arr1[0,1].set_xlim(grid[0], grid[-1])
                    arr1[1,1].set_xlim(grid[0], grid[-1])
                    arr1[0,1].legend()
                    arr1[1,1].legend()
                elif len(BS) < 8:
                    arr1[0,1].plot(grid, mf, 'x-k' ,label = 'HF')
                    arr1[0,1].plot(grid, mf_dg, 'x-r' ,label = 'HF (DG)')
                    arr1[1,1].plot(grid, mpe, 'x-k' ,label = 'MP2')
                    arr1[1,1].plot(grid, mpe_dg, 'x-r' ,label = 'MP2 (DG)')
                    arr1[1,1].set_title("6-311G ")#+ NDG + "/" + NBI)
                    arr1[0,1].set_xlim(grid[0], grid[-1])
                    arr1[1,1].set_xlim(grid[0], grid[-1])
                    arr1[0,1].legend()
                    arr1[1,1].legend()
            if '631' in BS and '11' not in BS:
                if len(BS) > 6:
                    arr1[0,2].plot(grid, mf, '+-k' ,label = 'HF (aug)')
                    arr1[0,2].plot(grid, mf_dg, '+-r' ,label = 'HF (aug-DG)')
                    arr1[1,2].plot(grid, mpe, '+-k' ,label = 'MP2 (aug)')
                    arr1[1,2].plot(grid, mpe_dg, '+-r' ,label = 'MP2 (aug-DG)')
                    arr1[0,2].set_title("6-31G ")# + NDG + "/" + NBI )
                    arr1[0,2].set_xlim(grid[0], grid[-1])
                    arr1[1,2].set_xlim(grid[0], grid[-1])
                    arr1[0,2].legend()
                    arr1[1,2].legend()
                elif len(BS) < 7:
                    arr1[0,2].plot(grid, mf, 'x-k' ,label = 'HF')
                    arr1[0,2].plot(grid, mf_dg, 'x-r' ,label = 'HF (DG)')
                    arr1[1,2].plot(grid, mpe, 'x-k' ,label = 'MP2')
                    arr1[1,2].plot(grid, mpe_dg, 'x-r' ,label = 'MP2 (DG)')
                    arr1[1,2].set_title("6-31G ")#+ NDG + "/" + NBI)
                    arr1[0,2].set_xlim(grid[0], grid[-1])
                    arr1[1,2].set_xlim(grid[0], grid[-1])
                    arr1[0,2].legend()
                    arr1[1,2].legend()
            if '321' in BS:
                if len(BS) > 6:
                    arr1[0,3].plot(grid, mf, '+-k' ,label = 'HF (aug)')
                    arr1[0,3].plot(grid, mf_dg, '+-r' ,label = 'HF (aug-DG)')
                    arr1[1,3].plot(grid, mpe, '+-k' ,label = 'MP2 (aug)')
                    arr1[1,3].plot(grid, mpe_dg, '+-r' ,label = 'MP2 (aug-DG)')
                    arr1[0,3].set_title("3-21G ")# + NDG + "/" + NBI )
                    arr1[0,3].set_xlim(grid[0], grid[-1])
                    arr1[1,3].set_xlim(grid[0], grid[-1])
                    arr1[0,3].legend()
                    arr1[1,3].legend()
                elif len(BS) < 7:
                    arr1[0,3].plot(grid, mf, 'x-k' ,label = 'HF')
                    arr1[0,3].plot(grid, mf_dg, 'x-r' ,label = 'HF (DG)')
                    arr1[1,3].plot(grid, mpe, 'x-k' ,label = 'MP2')
                    arr1[1,3].plot(grid, mpe_dg, 'x-r' ,label = 'MP2 (DG)')
                    arr1[1,3].set_title("3-21G ")#+ NDG + "/" + NBI)
                    arr1[0,3].set_xlim(grid[0], grid[-1])
                    arr1[1,3].set_xlim(grid[0], grid[-1])
                    arr1[0,3].legend()
                    arr1[1,3].legend()

            plt_b = False

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()


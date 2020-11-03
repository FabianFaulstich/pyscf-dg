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
    plt.rcParams.update({'font.size': 15})

    plt_b = False

    f   = open("out_grid_conv_leqOne.txt", "r")

    fig1, arr1 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))
    fig0, arr0 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))

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
           if '6311' in BS:
                if len(BS) > 7:
                    arr0.plot(grid, mf, '^-k' ,label = '6-311++G')
                    arr0.plot(grid, mf_dg, '^-r' ,label = 'DG-V-6-311++G')
                    arr1.plot(grid, mpe, '>-k' ,label = '6-311++G')
                    arr1.plot(grid, mpe_dg, '>-r' ,label = 'DG-V-6-311++G)')
                elif len(BS) < 8:
                    arr0.plot(grid, mf, 'v-k' ,label = '6-311G')
                    arr0.plot(grid, mf_dg, 'v-r' ,label = 'DG-6-311G')
                    arr1.plot(grid, mpe, '<-k' ,label = '6-311G')
                    arr1.plot(grid, mpe_dg, '<-r' ,label = 'DG-6-311G')
                
                arr0.set_xlim(grid[0], grid[-1])
                arr1.set_xlim(grid[0], grid[-1])
                arr0.set_ylabel('Energy (a.u.)')
                arr0.set_xlabel('Grid spacing (a.u.)')

                arr0.legend(frameon=False, loc = 'upper right')
                #arr1.legend(frameon=False, loc = 'upper right')
                arr0.set_xlim(grid[0], grid[-1])
                arr1.set_xlim(grid[0], grid[-1])
                arr1.ticklabel_format(axis = 'y',
                         style = 'sci',
                         scilimits = (2,2))
                arr1.set_ylabel('Energy (a.u.)')
                arr1.set_xlabel('Grid spacing (a.u.)')

                arr0.spines['top'].set_visible(False)
                arr0.spines['right'].set_visible(False)
                
                arr1.spines['top'].set_visible(False)
                arr1.spines['right'].set_visible(False)
            
           plt_b = False

    fig0.tight_layout()
    fig1.tight_layout()
    plt.show()


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

    ndg = False
    nbi = False
    bs = False
    mf = False
    mf_iter = False
    mf_dg_iter = False
    rt = False
    angle = False
    #angle_iter = False
    mf_plt = True
    con = False
    catch_cond = True

    f   = open("out_corr.txt", "r")
    cmap = get_cmp(9)

    fig0, ar0 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))
    fig1, ar1 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))
    fig2, ar2 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))
    fig3, ar3 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))

    for cnt, line in enumerate(f):

        if ndg:
            NDG = str(int(int(line)/4))
            ndg = False

        if nbi:
            NBI = line 
            nbi = False


        if bs:
            BS = line
            bs = False

        if angle:
            fl = [x for x in line.split()]
            fl[0] = fl[0][1:]
            fl[-1] = fl[-1][:-1]
            #fl = fl[1:] # neglecting first point
            if fl[-1] == "]":
                fl = fl[:-1]
            else:
                fl[-1] = fl[-1][:-1]
            rot_a = [float(x) for x in fl]
            angle = False


        if rt:
            relt = line
            rt = False

        if 'Rotation angles:' in line:
            angle = True

        if 'AO basis' in line:
            bs = True

        if 'Relative truncation' in line:
            rt = True

        if 'Number of DG basis functions:' in line:
            ndg = True

        if 'Number of Built in basis functions:' in line:
            nbi = True

        if 'Mean-field energy:' in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf = [float(x) for x in h]
            
            #plot
            if '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar0.plot(rot_a, mf, 'o-k' ,label = 'HF (6-31++G)')
                    ar0.set_ylim(-2.24,-2.05)
                    ar0.set_ylabel('Enegry (a.u.)')
                    ar0.spines['top'].set_visible(False)
                    ar0.spines['right'].set_visible(False)
                    ar0.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=2,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )

                elif len(BS) == 5:
                    ar1.plot(rot_a, mf, 'o-k' ,label = 'HF (6-31G)')
                    ar1.set_ylim(-2.24,-2.05)
                    ar1.spines['top'].set_visible(False)
                    ar1.spines['right'].set_visible(False)
                    ar1.set_ylabel('Enegry (a.u.)')
                    ar1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=2,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )

    
        if 'Mean-field energy (DG):' in line:
            fl = [x for x in line.split()]
            h  = fl[3:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_dg = [float(x) for x in h]
            
            #plot
            if '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar0.plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG-V)')
                    ar2.semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 
                            'o-b' ,label = 'HF')
                    ar2.set_ylim(1e-4, 5e-2)
                    ar2.set_ylabel('Enegry diff. ($E - E^{(DG-V)}$)')
                    ar2.set_xlabel('angle ($\\alpha$) in degree')
                    ar2.spines['top'].set_visible(False)
                    ar2.spines['right'].set_visible(False)
                    ar2.legend(frameon = False, loc = 4)
                elif len(BS) == 5:
                    ar1.plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG-V)')
                    ar1.legend()
                    ar3.semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 
                            'o-b' ,label = 'HF')
                    ar3.set_ylim(1e-4, 5e-2)
                    ar3.set_ylabel('Enegry diff. ($E - E^{(DG-V)}$)')
                    ar3.set_xlabel('angle ($\\alpha$) in degree')
                    ar3.spines['top'].set_visible(False)
                    ar3.spines['right'].set_visible(False)
                    ar3.legend(frameon = False, loc = 4)


        if 'MP2 corr. energy :' in line:
            fl = [x for x in line.split()]
            h  = fl[4:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            ecor = [float(x) for x in h]
            mp = [a + b for a,b in zip(mf,ecor)]
            #plot
            if '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar0.plot(rot_a, mp, 'v-k' ,label = 'MP2')
                elif len(BS) == 5:
                    ar1.plot(rot_a, mp, 'v-k' ,label = 'MP2')
                    ar1.legend(frameon = False, loc = 4)
    
        if 'MP2 corr. energy (DG) :' in line:
            fl = [x for x in line.split()]
            h  = fl[5:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            ecor = [float(x) for x in h]
            mp_dg = [a + b for a,b in zip(mf_dg,ecor)]
            
            #plot
            if '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar0.plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG-V)')
                    ar2.semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    ar2.legend(frameon= False, loc = 4)
                elif len(BS) == 5:
                    ar1.plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG-V)')
                    ar1.legend()
                    ar3.semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    ar3.legend(frameon = False, loc = 4)
        
        if 'CCSD corr. energy:' in line:
            fl = [x for x in line.split()]
            h  = fl[3:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            ecor = [float(x) for x in h]
            cc = [a + b for a,b in zip(mf,ecor)]
            #plot
            if '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar0.plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                elif len(BS) == 5:
                    ar1.plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                    ar1.legend()
    
        if 'CCSD corr. energy (DG):' in line:
            fl = [x for x in line.split()]
            h  = fl[4:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            ecor = [float(x) for x in h]
            cc_dg = [a + b for a,b in zip(mf_dg,ecor)]
            #plot
            if '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar0.plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG-V)')
                    ar2.semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    ar2.legend()
                elif len(BS) == 5:
                    ar1.plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG-V)')
                    ar1.legend()
                    ar3.semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    ar3.legend()


        if 'AO basis:' in line:
            catch_cond = True
            dgc = 0
   
    ar0.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=2,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )
    ar1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=2,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )
    ar2.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=3,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )

    ar3.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=3,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )

    ar0.set_xlabel('angle ($\\alpha$) in degree')
    ar1.set_xlabel('angle ($\\alpha$) in degree')

    fig0.tight_layout()
    fig0.subplots_adjust(top=0.76, bottom = .15)
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.76, bottom = .15)
    fig2.tight_layout()
    fig3.tight_layout()
    fig2.subplots_adjust(top=0.92, bottom = .15)
    fig3.subplots_adjust(top=0.92, bottom = .15)
    plt.show()

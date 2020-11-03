import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

import numpy as np
from scipy.optimize import curve_fit

def func(x,Y,a):
    return Y + a * x**(-3)

def get_cmp(n, name = 'hsv'):
    return plt.cm.get_cmap(name, n)

if __name__ == '__main__':

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"    
    plt.rcParams.update({'font.size': 16})

    bs = False
    mf = False
    mf_iter = False
    mf_dg_iter = False
    rt = False
    angle = False
    angle_iter = False
    mf_plt = True
    con = False
    catch_cond = True

    f   = open("out_tr.txt", "r")
    cmap = get_cmp(9)

    fig0, ar0 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))
    fig1, ar1 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))
    fig2, ar2 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))
    fig3, ar3 = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))

    for cnt, line in enumerate(f):

        if bs:
            BS = line
            bs = False
        
        if angle_iter:
            angle_iter = False
            fl = [x for x in line.split()]
            fl[-1] = fl[-1][:-1]
            rot_a += [180.0/np.pi*float(fl[0])]

        if angle:
            fl = [x for x in line.split()]
            #fl[0] = fl[0][1:]
            fl = fl[1:] # neglecting first point
            if fl[-1] == "]":
                fl = fl[:-1]
            else:
                fl[-1] = fl[-1][:-1]
            rot_a = [180.0*float(x)/np.pi for x in fl]
            angle = False
            angle_iter = True
        
        if rt:
            relt = line
            rt = False

        if 'Rotation angles:' in line:
            angle = True
        
        if 'AO basis' in line:
            bs = True

        if 'Relative truncation' in line:
            rt = True

        if mf_iter:
            mf_iter = False
            fl = [x for x in line.split()]
            fl[-1] = fl[-1][:-1]
            mf += [float(fl[0])] 
            if mf_plt:
                #plot
                if 'ccpvdz' in BS:
                    if len(BS) > 7:
                        ar0.plot(rot_a,mf[1:],'D-k', label = 'aug-cc-pVDZ')
                        ar0.set_ylim(-2.17,-2.06)
                        ar0.set_ylabel('Enegry (a.u.)')
                        ar0.xaxis.set_ticks([15,30,45,60,75,90])
                        ar0.set_xlabel('Angle ($\\alpha$) in degree')
                        ar0.spines['top'].set_visible(False)
                        ar0.spines['right'].set_visible(False)
                        fig0.tight_layout()
                        ar0.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
                                   ncol=2, 
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   ) 
                    elif len(BS) < 8:
                        ar1.plot(rot_a,mf[1:],'D-k',label = 'cc-pVDZ')
                        ar1.legend(loc='upper right', ncol=2)
                        ar1.set_ylabel('Enegry (a.u.)')
                        ar1.set_ylim(-2.17,-2.06)
                        ar1.xaxis.set_ticks([15,30,45,60,75,90])
                        ar1.set_xlabel('Angle ($\\alpha$) in degree')
                        ar1.spines['top'].set_visible(False)
                        ar1.spines['right'].set_visible(False)
                        fig1.tight_layout()
                        ar1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                               ncol=2,
                               frameon=False,
                               mode = 'expand', 
                               borderaxespad=0.)
                mf_plt = False
       
        if 'Mean-field energy:' in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf = [float(x) for x in h]
            mf_iter = True

        
        if mf_dg_iter and '0.5' not in relt:
            mf_dg_iter = False
            fl = [x for x in line.split()]
            fl[-1] = fl[-1][:-1]
            mf_dg += [float(fl[0])]
            diff = np.array([a-b for a,b in zip(mf[1:],mf_dg[1:])])
            mini=np.amin(diff)
            mini=1000*np.round(mini,4)
            maxi=np.amax(diff)
            maxi=1000*np.round(maxi,4)
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    rel = float(relt)
                    rel *= 100
                    rel = str(int(rel))
                    label_str = 'DG-V (' + rel + '%)'
                    ar0.plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x',label = label_str)
                    ar2.plot(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    ar2.set_ylim(1e-4,1e-3)
                    ar2.set_ylabel('Enegry diff. ($E_{HF} - E_{HF}^{(DG-V)}$)')
                    ar2.set_xlabel('Angle ($\\alpha$) in degree')
                    ar2.xaxis.set_ticks([15,30,45,60,75,90])
                    ar2.spines['top'].set_visible(False)
                    ar2.spines['right'].set_visible(False)
                    fig2.tight_layout()
                    ar0.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                               ncol=2,
                               frameon=False,
                               mode = 'expand', 
                               borderaxespad=0.)
                    fig0.tight_layout()
                elif len(BS) < 8:
                    rel = float(relt)
                    rel *= 100
                    rel = str(int(rel))
                    label_str = 'DG-V (' + rel + '%)'
                    ar1.plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x',label = label_str)
                    ar3.plot(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    ar3.set_ylim(1e-4,1e-3)
                    ar3.set_ylabel('Enegry diff. ($E_{HF} - E_{HF}^{(DG-V)}$)')
                    
                    ar3.set_xlabel('Angle ($\\alpha$) in degree')
                    ar3.xaxis.set_ticks([15,30,45,60,75,90])
                    ar3.spines['top'].set_visible(False)
                    ar3.spines['right'].set_visible(False)
                    fig3.tight_layout()
                    ar1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                               ncol=2,
                               frameon=False,
                               mode = 'expand', 
                               borderaxespad=0.) 

        if 'Mean-field energy (DG):' in line and '0.5' not in relt:
            fl = [x for x in line.split()]
            h  = fl[3:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_dg = [float(x) for x in h]
            dgc += 1
            mf_dg_iter = True

        if 'AO basis:' in line:
            mf_plt = True
            catch_cond = True
            dgc = 0
    
    ar3.set_yscale('log')
    ar3.set_yticks([1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3]) 
    ar3.yaxis.set_minor_locator(mtick.NullLocator())
    ar3.get_yaxis().get_major_formatter().labelOnlyBase = False
    ar3.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ar3.ticklabel_format(axis = 'y',
                         style = 'sci',
                         scilimits = (4,4))
    
    ar2.set_yscale('log')
    ar2.set_yticks([1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3])                  
    ar2.yaxis.set_minor_locator(mtick.NullLocator())
    ar2.get_yaxis().get_major_formatter().labelOnlyBase = False
    ar2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ar2.ticklabel_format(axis = 'y',
                         style = 'sci',
                         scilimits = (4,4))
    
    
    fig0.subplots_adjust(top=0.66, bottom = .16)
    fig1.subplots_adjust(top=0.66, bottom = .16)
    plt.show()

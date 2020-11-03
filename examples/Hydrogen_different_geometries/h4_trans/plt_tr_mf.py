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

    fig, arccpvdz = plt.subplots(nrows=2, ncols=2,  figsize=(12,5.5))
    fig1, ar6311  = plt.subplots(nrows=2, ncols=2,  figsize=(12,5.5))
    fig2, ar631   = plt.subplots(nrows=2, ncols=2,  figsize=(12,5.5))
    fig3, ar321   = plt.subplots(nrows=2, ncols=2,  figsize=(12,5.5))

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
                        arccpvdz[0,0].plot(rot_a,mf[1:],'D-k',
                                label = 'aug-cc-pVDZ')
                        arccpvdz[0,0].set_ylim(-2.17,-2.06)
                        arccpvdz[0,0].set_ylabel('Enegry (a.u.)')
                        arccpvdz[0,0].xaxis.set_ticks([15,30,45,60,75,90])
                        arccpvdz[0,0].xaxis.set_ticklabels([])
                        arccpvdz[0,0].spines['top'].set_visible(False)
                        arccpvdz[0,0].spines['right'].set_visible(False)
                        arccpvdz[0,0].legend(loc='upper right', ncol=2, 
                                frameon=False) 
                    elif len(BS) < 8:
                        arccpvdz[0,1].plot(rot_a,mf[1:],'D-k',label = 'cc-pVDZ')
                        arccpvdz[0,1].legend(loc='upper right', ncol=2)
                        arccpvdz[0,1].set_ylim(-2.17,-2.06)
                        #arccpvdz[0,1].set_ylabel('Enegry (a.u.)')
                        arccpvdz[0,1].xaxis.set_ticks([15,30,45,60,75,90])
                        arccpvdz[0,1].xaxis.set_ticklabels([])
                        #arccpvdz[0,1].yaxis.set_ticklabels([])
                        arccpvdz[0,1].spines['top'].set_visible(False)
                        arccpvdz[0,1].spines['right'].set_visible(False)
                        arccpvdz[0,1].legend(loc='upper right', ncol=2
                                , frameon=False)

                elif '6311' in BS:
                    if len(BS) == 8:
                        ar6311[0,0].plot(rot_a, mf[1:], 'D-k' ,label = '6-311++G')
                        ar6311[0,0].set_ylim(-2.17,-2.06)
                        ar6311[0,0].set_ylabel('Enegry (a.u.)')
                        ar6311[0,0].xaxis.set_ticks([15,30,45,60,75,90])
                        ar6311[0,0].xaxis.set_ticklabels([])
                        ar6311[0,0].spines['top'].set_visible(False)
                        ar6311[0,0].spines['right'].set_visible(False)
                        ar6311[0,0].legend(loc='upper right', ncol=2,
                                frameon=False)
                    elif len(BS) == 6:
                        ar6311[0,1].plot(rot_a, mf[1:], 'D-k' ,label = '6-311G')
                        ar6311[0,1].legend(loc='upper right', ncol=2)
                        ar6311[0,1].set_ylim(-2.17,-2.06)
                        #arccpvdz[0,1].set_ylabel('Enegry (a.u.)')
                        ar6311[0,1].xaxis.set_ticks([15,30,45,60,75,90])
                        ar6311[0,1].xaxis.set_ticklabels([])
                        #arccpvdz[0,1].yaxis.set_ticklabels([])
                        ar6311[0,1].spines['top'].set_visible(False)
                        ar6311[0,1].spines['right'].set_visible(False)
                        ar6311[0,1].legend(loc='upper right', ncol=2)
                elif '631' in BS and '11' not in BS:
                    if len(BS) == 7:
                        ar631[0,0].plot(rot_a, mf[1:], 'D-k' ,label = '6-31++G')
                        ar631[0,0].set_ylim(-2.17,-2.02)
                        ar631[0,0].set_ylabel('Enegry (a.u.)')
                        ar631[0,0].xaxis.set_ticks([15,30,45,60,75,90])
                        ar631[0,0].xaxis.set_ticklabels([])
                        ar631[0,0].spines['top'].set_visible(False)
                        ar631[0,0].spines['right'].set_visible(False)
                        ar631[0,0].legend(loc='upper right', ncol=2,
                                frameon=False)
                    elif len(BS) == 5:
                        ar631[0,1].plot(rot_a, mf[1:], 'D-k' ,label = '6-31G')
                        ar631[0,1].set_ylim(-2.17,-2.02)
                        #arccpvdz[0,1].set_ylabel('Enegry (a.u.)')
                        ar631[0,1].xaxis.set_ticks([15,30,45,60,75,90])
                        ar631[0,1].xaxis.set_ticklabels([])
                        #arccpvdz[0,1].yaxis.set_ticklabels([])
                        ar631[0,1].spines['top'].set_visible(False)
                        ar631[0,1].spines['right'].set_visible(False)
                        ar631[0,1].legend(loc='upper right', ncol=2)

                elif '321' in BS:
                    if len(BS) == 7:
                        ar321[0,0].plot(rot_a, mf[1:], 'D-k' ,label ='3-21++G')
                        ar321[0,0].legend(loc='upper right', ncol=2)
                        ar321[0,0].set_ylim(-2.17,-2.02)
                        ar321[0,0].set_ylabel('Enegry (a.u.)')
                        ar321[0,0].xaxis.set_ticks([15,30,45,60,75,90])
                        ar321[0,0].xaxis.set_ticklabels([])
                        ar321[0,0].spines['top'].set_visible(False)
                        ar321[0,0].spines['right'].set_visible(False)
                        ar321[0,0].legend(loc='upper right', ncol=2,
                                frameon=False)
                    elif len(BS) == 5:
                        ar321[0,1].plot(rot_a, mf[1:], 'D-k' ,label = '3-21G')
                        ar321[0,1].legend(loc='upper right', ncol=2) 
                        ar321[0,1].set_ylim(-2.17,-2.02)
                        #arccpvdz[0,1].set_ylabel('Enegry (a.u.)')
                        ar321[0,1].xaxis.set_ticks([15,30,45,60,75,90])
                        ar321[0,1].xaxis.set_ticklabels([])
                        #arccpvdz[0,1].yaxis.set_ticklabels([])
                        ar321[0,1].spines['top'].set_visible(False)
                        ar321[0,1].spines['right'].set_visible(False)
                        ar321[0,1].legend(loc='upper right', ncol=2)
            

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
                    arccpvdz[0,0].plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x',label = "DG-V-" + relt)
                    arccpvdz[1,0].semilogy(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    arccpvdz[1,0].set_ylim(1e-4,1e-3)
                    arccpvdz[1,0].set_ylabel('Enegry difference ($E_{HF} - E_{HF}^{(DG-V)}$)')
                    arccpvdz[1,0].set_xlabel('angle ($\\alpha$) in degree')
                    arccpvdz[1,0].xaxis.set_ticks([15,30,45,60,75,90])
                    arccpvdz[1,0].spines['top'].set_visible(False)
                    arccpvdz[1,0].spines['right'].set_visible(False)
                    arccpvdz[0,0].legend(loc='upper right', ncol=2) 
                elif len(BS) < 8:
                    arccpvdz[0,1].plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x',label = "DG-V-" + relt)
                    arccpvdz[1,1].semilogy(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    arccpvdz[1,1].set_ylim(1e-4,1e-3)
                    #arccpvdz[1,1].set_ylabel('Enegry (a.u.)')
                    arccpvdz[1,1].set_xlabel('angle ($\\alpha$) in degree')
                    #arccpvdz[1,1].yaxis.set_ticks([1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,
                    #    7e-4,8e-4,9e-4,1e-3])
                    #arccpvdz[1,1].yaxis.set_ticklabels([])
                    arccpvdz[1,1].xaxis.set_ticks([15,30,45,60,75,90])
                    arccpvdz[1,1].spines['top'].set_visible(False)
                    arccpvdz[1,1].spines['right'].set_visible(False)
                    arccpvdz[0,1].legend(loc='upper right', ncol=2) 

            elif '6311' in BS: 
                if len(BS) == 8:
                    ar6311[0,0].plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x' ,label =  "DG-V-" + relt) 
                    ar6311[1,0].semilogy(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    ar6311[1,0].set_ylabel('Enegry difference ($E_{HF} - E_{HF}^{(DG-V)}$)')
                    ar6311[1,0].set_xlabel('angle ($\\alpha$) in degree')
                    ar6311[0,0].legend(loc='upper right', ncol=2) 
                elif len(BS) == 6:
                    ar6311[0,1].plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x' ,label =   "DG-V-" + relt)
                    ar6311[1,1].semilogy(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    ar6311[1,1].set_xlabel('angle ($\\alpha$) in degree')
                    ar6311[0,1].legend(loc='upper right', ncol=2) 
            
            elif '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar631[0,0].plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x' ,label =  "DG-V-" + relt)
                    ar631[1,0].semilogy(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    ar631[1,0].set_ylabel('Enegry difference ($E_{HF} - E_{HF}^{(DG-V)}$)')
                    ar631[1,0].set_xlabel('angle ($\\alpha$) in degree')
                    ar631[0,0].legend(loc='upper right', ncol=2) 
                elif len(BS) == 5:
                    ar631[0,1].plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x' ,label =   "DG-V-" + relt)
                    ar631[1,1].semilogy(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    ar631[1,1].set_xlabel('angle ($\\alpha$) in degree')
                    ar631[0,1].legend(loc='upper right', ncol=2) 

            elif '321' in BS:
                if len(BS) == 7:
                    ar321[0,0].plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x' ,label =   "DG-V-" + relt)
                    ar321[1,0].semilogy(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    ar321[1,0].set_ylabel('Enegry difference ($E_{HF} - E_{HF}^{(DG-V)}$)')
                    ar321[1,0].set_xlabel('angle ($\\alpha$) in degree')
                    ar321[0,0].legend(loc='upper right', ncol=2)
                elif len(BS) == 5:
                    ar321[0,1].plot(rot_a, mf_dg[1:], color = cmap(dgc), 
                            marker= 'x' ,label =   "DG-V-" + relt)
                    ar321[1,1].semilogy(rot_a, [a-b for a,b in zip(mf[1:],
                        mf_dg[1:])], color = cmap(dgc), marker= 'x',
                        label = "DG-V-" + relt)
                    ar321[1,1].set_xlabel('angle ($\\alpha$) in degree')
                    ar321[0,1].legend(loc='upper right', ncol=2)

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
    

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.95)
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.95)
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.95)
    plt.show()

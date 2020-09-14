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

    f   = open("out.txt", "r")
    cmap = get_cmp(9)

    fig, arccpvdz = plt.subplots(nrows=2, ncols=2,  figsize=(10,8))
    fig1, ar6311  = plt.subplots(nrows=2, ncols=2,  figsize=(10,8))
    fig2, ar631   = plt.subplots(nrows=2, ncols=2,  figsize=(10,8))
    fig3, ar321   = plt.subplots(nrows=2, ncols=2,  figsize=(10,8))

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

        #if angle_iter:
        #    angle_iter = False
        #    fl = [x for x in line.split()]
        #    fl[-1] = fl[-1][:-1]
        #    rot_a += [float(fl[0])]
        
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
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    arccpvdz[0,0].plot(rot_a, mf, 'o-k' ,label = 'HF')
                    arccpvdz[0,0].set_title("aug-cc-pVDZ ")# + NDG + "/" + NBI )
                    arccpvdz[0,0].legend()
                elif len(BS) < 8:
                    arccpvdz[0,1].plot(rot_a, mf, 'o-k' ,label = 'HF')
                    arccpvdz[0,1].set_title("cc-pVDZ ")#+ NDG + "/" + NBI)
                    arccpvdz[0,1].legend()
            elif '6311' in BS:
                if len(BS) == 8:
                    ar6311[0,0].plot(rot_a, mf, 'o-k' ,label = 'HF')
                    ar6311[0,0].set_title("6-311++G ")#+ NDG + "/" + NBI)
                    ar6311[0,0].legend()
                #elif len(BS) == 7:
                #    ar6311[0,1].plot(rot_a, mf, 'o-k' ,label = 'HF')
                #    ar6311[0,1].set_title("6-311+G "+ NDG + "/" + NBI)
                #    ar6311[0,1].legend()
                elif len(BS) == 6:
                    ar6311[0,1].plot(rot_a, mf, 'o-k' ,label = 'HF')
                    ar6311[0,1].set_title("6-311G ")#+ NDG + "/" + NBI)
                    ar6311[0,1].legend()
            elif '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar631[0,0].plot(rot_a, mf, 'o-k' ,label = 'HF')
                    ar631[0,0].set_title("6-31++G ")#+ NDG + "/" + NBI)
                    ar631[0,0].legend()
                #elif len(BS) == 6:
                #    ar631[0,1].plot(rot_a, mf, 'o-k' ,label = 'HF')
                #    ar631[0,1].set_title("6-31+G "+ NDG + "/" + NBI)
                #    ar631[0,1].legend()
                elif len(BS) == 5:
                    ar631[0,1].plot(rot_a, mf, 'o-k' ,label = 'HF')
                    ar631[0,1].set_title("6-31G ")#+ NDG + "/" + NBI)
                    ar631[0,1].legend()
            elif '321' in BS:
                if len(BS) == 7:
                    ar321[0,0].plot(rot_a, mf, 'o-k' ,label = 'HF')
                    ar321[0,0].set_title("3-21++G ")#+ NDG + "/" + NBI)
                    ar321[0,0].legend()
                elif len(BS) == 5:
                    ar321[0,1].plot(rot_a, mf, 'o-k' ,label = 'HF')
                    ar321[0,1].set_title("3-21G ")#+ NDG + "/" + NBI)
                    ar321[0,1].legend()
    
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
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    arccpvdz[0,0].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                    #arccpvdz[0,0].set_title("aug-cc-pVDZ")
                    arccpvdz[0,0].legend()
                    arccpvdz[1,0].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                    arccpvdz[1,0].legend()
                elif len(BS) < 8:
                    arccpvdz[0,1].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                    #arccpvdz[0,1].set_title("cc-pVDZ")
                    arccpvdz[1,1].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                    arccpvdz[1,1].legend()
                    arccpvdz[0,1].legend()
            elif '6311' in BS:
                if len(BS) == 8:
                    ar6311[0,0].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                    #ar6311[0,0].set_title("6-311++G")
                    ar6311[0,0].legend()
                    ar6311[1,0].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                    ar6311[1,0].legend()
                #elif len(BS) == 7:
                #    ar6311[0,1].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                #    #ar6311[0,1].set_title("6-311+G")
                #    ar6311[0,1].legend()
                #    ar6311[1,1].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                #    ar6311[1,1].legend()
                elif len(BS) == 6:
                    ar6311[0,1].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                    #ar6311[0,1].set_title("6-311G")
                    ar6311[0,1].legend()
                    ar6311[1,1].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                    ar6311[1,1].legend()
            elif '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar631[0,0].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                    #ar631[0,0].set_title("6-31++G")
                    ar631[0,0].legend()
                    ar631[1,0].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                    ar631[1,0].legend()
                #elif len(BS) == 6:
                #    ar631[0,1].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                #    #ar631[0,1].set_title("6-31+G")
                #    ar631[0,1].legend()
                #    ar631[1,1].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                #    ar631[1,1].legend()
                elif len(BS) == 5:
                    ar631[0,1].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                    #ar631[0,1].set_title("6-31G")
                    ar631[0,1].legend()
                    ar631[1,1].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                    ar631[1,1].legend()
            elif '321' in BS:
                if len(BS) == 7:
                    ar321[0,0].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                    #ar321[0,0].set_title("3-21++G")
                    ar321[0,0].legend()
                    ar321[1,0].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                    ar321[1,0].legend()
                elif len(BS) == 5:
                    ar321[0,1].plot(rot_a, mf_dg, 'o-b' ,label = 'HF (DG)')
                    #ar321[0,1].set_title("3-21G")
                    ar321[0,1].legend()
                    ar321[1,1].semilogy(rot_a, [a-b for a,b in zip(mf,mf_dg)], 'o-b' ,label = 'HF')
                    ar321[1,1].legend()


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
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    arccpvdz[0,0].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                    #arccpvdz[0,0].set_title("aug-cc-pVDZ")
                    arccpvdz[0,0].legend()
                elif len(BS) < 8:
                    arccpvdz[0,1].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                    #arccpvdz[0,1].set_title("cc-pVDZ")
                    arccpvdz[0,1].legend()
            elif '6311' in BS:
                if len(BS) == 8:
                    ar6311[0,0].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                    #ar6311[0,0].set_title("6-311++G")
                    ar6311[0,0].legend()
                #elif len(BS) == 7:
                #    ar6311[0,1].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                #    #ar6311[0,1].set_title("6-311+G")
                #    ar6311[0,1].legend()
                elif len(BS) == 6:
                    ar6311[0,1].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                    #ar6311[0,1].set_title("6-311G")
                    ar6311[0,1].legend()
            elif '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar631[0,0].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                    #ar631[0,0].set_title("6-31++G")
                    ar631[0,0].legend()
                #elif len(BS) == 6:
                #    ar631[0,1].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                #    #ar631[0,1].set_title("6-31+G")
                #    ar631[0,1].legend()
                elif len(BS) == 5:
                    ar631[0,1].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                    #ar631[0,1].set_title("6-31G")
                    ar631[0,1].legend()
            elif '321' in BS:
                if len(BS) == 7:
                    ar321[0,0].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                    #ar321[0,0].set_title("3-21++G")
                    ar321[0,0].legend()
                elif len(BS) == 5:
                    ar321[0,1].plot(rot_a, mp, 'v-k' ,label = 'MP2')
                    #ar321[0,1].set_title("3-21G")
                    ar321[0,1].legend()
    
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
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    arccpvdz[0,0].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                    #arccpvdz[0,0].set_title("aug-cc-pVDZ")
                    arccpvdz[0,0].legend()
                    arccpvdz[1,0].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    arccpvdz[1,0].legend()
                elif len(BS) < 8:
                    arccpvdz[0,1].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                    #arccpvdz[0,1].set_title("cc-pVDZ")
                    arccpvdz[0,1].legend()
                    arccpvdz[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    arccpvdz[1,1].legend()
            elif '6311' in BS:
                if len(BS) == 8:
                    ar6311[0,0].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                    #ar6311[0,0].set_title("6-311++G")
                    ar6311[0,0].legend()
                    ar6311[1,0].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    ar6311[1,0].legend()
                #elif len(BS) == 7:
                #    ar6311[0,1].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                #    #ar6311[0,1].set_title("6-311+G")
                #    ar6311[0,1].legend()
                #    ar6311[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                #    ar6311[1,1].legend()
                elif len(BS) == 6:
                    ar6311[0,1].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                    #ar6311[0,1].set_title("6-311G")
                    ar6311[0,1].legend()
                    ar6311[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    ar6311[1,1].legend()
            elif '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar631[0,0].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                    #ar631[0,0].set_title("6-31++G")
                    ar631[0,0].legend()
                    ar631[1,0].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    ar631[1,0].legend()
                #elif len(BS) == 6:
                #    ar631[0,1].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                #    #ar631[0,1].set_title("6-31+G")
                #    ar631[0,1].legend()
                #    ar631[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                #    ar631[1,1].legend()
                elif len(BS) == 5:
                    ar631[0,1].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                    #ar631[0,1].set_title("6-31G")
                    ar631[0,1].legend()
                    ar631[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    ar631[1,1].legend()
            elif '321' in BS:
                if len(BS) == 7:
                    ar321[0,0].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                    #ar321[0,0].set_title("3-21++G")
                    ar321[0,0].legend()
                    ar321[1,0].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    ar321[1,0].legend()
                elif len(BS) == 5:
                    ar321[0,1].plot(rot_a, mp_dg, 'v-b' ,label = 'MP2 (DG)')
                    #ar321[0,1].set_title("3-21G")
                    ar321[0,1].legend()
                    ar321[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'v-b' ,label = 'MP2')
                    ar321[1,1].legend()
        
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
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    arccpvdz[0,0].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                    #arccpvdz[0,0].set_title("aug-cc-pVDZ")
                    arccpvdz[0,0].legend()
                elif len(BS) < 8:
                    arccpvdz[0,1].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                    #arccpvdz[0,1].set_title("cc-pVDZ")
                    arccpvdz[0,1].legend()
            elif '6311' in BS:
                if len(BS) == 8:
                    ar6311[0,0].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                    #ar6311[0,0].set_title("6-311++G")
                    ar6311[0,0].legend()
                #elif len(BS) == 7:
                #    ar6311[0,1].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                #    #ar6311[0,1].set_title("6-311+G")
                #    ar6311[0,1].legend()
                elif len(BS) == 6:
                    ar6311[0,1].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                    #ar6311[0,1].set_title("6-311G")
                    ar6311[0,1].legend()
            elif '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar631[0,0].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                    #ar631[0,0].set_title("6-31++G")
                    ar631[0,0].legend()
                #elif len(BS) == 6:
                #    ar631[0,1].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                #    #ar631[0,1].set_title("6-31+G")
                #    ar631[0,1].legend()
                elif len(BS) == 5:
                    ar631[0,1].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                    #ar631[0,1].set_title("6-31G")
                    ar631[0,1].legend()
            elif '321' in BS:
                if len(BS) == 7:
                    ar321[0,0].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                    #ar321[0,0].set_title("3-21++G")
                    ar321[0,0].legend()
                elif len(BS) == 5:
                    ar321[0,1].plot(rot_a, cc, 'x-k' ,label = 'CCSD')
                    #ar321[0,1].set_title("3-21G")
                    ar321[0,1].legend()
    
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
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    arccpvdz[0,0].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                    #arccpvdz[0,0].set_title("aug-cc-pVDZ")
                    arccpvdz[0,0].legend()
                    arccpvdz[1,0].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    arccpvdz[1,0].legend()
                elif len(BS) < 8:
                    arccpvdz[0,1].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                    #arccpvdz[0,1].set_title("cc-pVDZ")
                    arccpvdz[0,1].legend()
                    arccpvdz[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    arccpvdz[1,1].legend()
            elif '6311' in BS:
                if len(BS) == 8:
                    ar6311[0,0].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                    #ar6311[0,0].set_title("6-311++G")
                    ar6311[0,0].legend()
                    ar6311[1,0].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    ar6311[1,0].legend()
                #elif len(BS) == 7:
                #    ar6311[0,1].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                #    #ar6311[0,1].set_title("6-311+G")
                #    ar6311[0,1].legend()
                #    ar6311[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                #    ar6311[1,1].legend()
                elif len(BS) == 6:
                    ar6311[0,1].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                    #ar6311[0,1].set_title("6-311G")
                    ar6311[0,1].legend()
                    ar6311[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    ar6311[1,1].legend()
            elif '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar631[0,0].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                    #ar631[0,0].set_title("6-31++G")
                    ar631[0,0].legend()
                    ar631[1,0].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    ar631[1,0].legend()
                #elif len(BS) == 6:
                #    ar631[0,1].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                #    #ar631[0,1].set_title("6-31+G")
                #    ar631[0,1].legend()
                #    ar631[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                #    ar631[1,1].legend()
                elif len(BS) == 5:
                    ar631[0,1].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                    #ar631[0,1].set_title("6-31G")
                    ar631[0,1].legend()
                    ar631[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    ar631[1,1].legend()
            elif '321' in BS:
                if len(BS) == 7:
                    ar321[0,0].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                    #ar321[0,0].set_title("3-21++G")
                    ar321[0,0].legend()
                    ar321[1,0].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    ar321[1,0].legend()
                elif len(BS) == 5:
                    ar321[0,1].plot(rot_a, cc_dg, 'x-b' ,label = 'CCSD (DG)')
                    #ar321[0,1].set_title("3-21G")
                    ar321[0,1].legend()
                    ar321[1,1].semilogy(rot_a, [a-b for a,b in zip(mp,mp_dg)], 'x-b' ,label = 'CCSD')
                    ar321[1,1].legend()




        if 'AO basis:' in line:
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

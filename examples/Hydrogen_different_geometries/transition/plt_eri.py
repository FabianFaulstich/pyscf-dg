import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats


def func(x, a,c):
    return  c*x**a 

def func1(x, a, c):
    return  a*x + c

if __name__ == '__main__':

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    f = open("out_tol1e-3_op.txt", "r")
    g = open("out_tol1e-2_op.txt", "r")
    j = open("out_tol1e-1_op.txt", "r")

    fig1, arr1 = plt.subplots(nrows=1, ncols=3,  figsize=(15,6))
    fig2, arr2 = plt.subplots(nrows=1, ncols=1,  figsize=(5,5))
    fig3, arr3 = plt.subplots(nrows=1, ncols=1,  figsize=(5,5))
    fig4, arr4 = plt.subplots(nrows=1, ncols=1,  figsize=(7,4))
    
    atoms = np.linspace(2, 30, 15, dtype = int)

    nnz_b    = False
    nnz_dg_b = False
    nnz_pw_b = False
    la_b     = False
    la_dg_b  = False
    n_ao_b   = False
    n_ao_dg_b = False

    
    nnz_eri     = []
    nnz_eri_pw  = []
    nnz_eri_dg  = []
    nnz_eri_dg1 = []
    nnz_eri_dg2 = []
    la          = []
    la_dg       = []
    la_dg1      = []
    la_dg2      = []


    for cnt, line in enumerate(j):

        if n_ao_dg_b:
            if "nnz_eri" in line:
                n_ao_dg_b = False
            else:
                hf = line.split()
                if hf[0][0] == '[':
                    if hf[0] == '[':
                        hf = hf[1:]
                    else:
                        hf[0] = hf[0][1:]
                elif hf[-1][-1] == ']':
                    hf[-1] = hf[-1][:-1]
                nao_dg1 += [float(x) for x in hf]

        if "Number of AO's (DG):" in line:
            n_ao_dg_b = True
            nao_dg1 = []

        if n_ao_b:
            if "nnz_eri" in line:
                n_ao_b = False
            else:
                hf = line.split()
                if hf[0][0] == '[':
                    if hf[0] == '[':
                        hf = hf[1:]
                    else:
                        hf[0] = hf[0][1:]
                elif hf[-1][-1] == ']':
                    hf[-1] = hf[-1][:-1]
                nao += [float(x) for x in hf]

        if "Number of AO's:" in line:
            n_ao_b = True
            nao = []

        if la_dg_b:
            fl = [x for x in line.split()]
            if fl[0][0] == '[':
                    if fl[0] == '[':
                        fl = fl[1:]
                    else:
                        fl[0] = fl[0][1:]
            elif fl[-1][-1] == ']':
                fl[-1] = fl[-1][:-1]
                la_dg_b = False
            la_dg2 += [float(x) for x in fl]

        if nnz_dg_b:
            if 'Lambda' in line:
                nnz_dg_b = False
                la_dg_b  = True
            else:
                fl = [x for x in line.split()]
                if fl[0][0] == '[':
                    if fl[0] == '[':
                        fl = fl[1:]
                    else:
                        fl[0] = fl[0][1:]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                nnz_eri_dg2 += [float(x) for x in fl]

        if 'nnz_eri (DG):' in line:
            nnz_dg_b = True
    

    for cnt, line in enumerate(g):
        
        if n_ao_dg_b:
            if "nnz_eri" in line:
                n_ao_dg_b = False
            else:
                hf = line.split()
                if hf[0][0] == '[':
                    if hf[0] == '[':
                        hf = hf[1:]
                    else:
                        hf[0] = hf[0][1:]
                elif hf[-1][-1] == ']':
                    hf[-1] = hf[-1][:-1]
                nao_dg2 += [float(x) for x in hf]

        if "Number of AO's (DG):" in line:
            n_ao_dg_b = True
            nao_dg2 = []

        if la_dg_b:
            if 'nnz_eri (PW):' in line:
                la_dg_b  = False
            else:
                fl = [x for x in line.split()]
                if fl[0][0] == '[':
                    if fl[0] == '[':
                        fl = fl[1:]
                    else:
                        fl[0] = fl[0][1:]
                elif fl[-1][-1] == ']':
                    if fl[-1] == ']':
                        fl = fl[:-1]
                    else:
                        fl[-1] = fl[-1][:-1]
                la_dg1 += [float(x) for x in fl]

        if nnz_dg_b:
            if 'Lambda' in line:
                nnz_dg_b = False
                la_dg_b  = True
            else:
                fl = [x for x in line.split()]
                if fl[0][0] == '[':
                    if fl[0] == '[':
                        fl = fl[1:]
                    else:
                        fl[0] = fl[0][1:]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                nnz_eri_dg1 += [float(x) for x in fl]

        if 'nnz_eri (DG):' in line:
            nnz_dg_b = True
        
    for cnt, line in enumerate(f):
 
        if n_ao_dg_b:
            if "nnz_eri" in line:
                n_ao_dg_b = False
            else:
                hf = line.split()
                if hf[0][0] == '[':
                    if hf[0] == '[':
                        hf = hf[1:]
                    else:
                        hf[0] = hf[0][1:]
                elif hf[-1][-1] == ']':
                    hf[-1] = hf[-1][:-1]
                nao_dg3 += [float(x) for x in hf]

        if "Number of AO's (DG):" in line:
            n_ao_dg_b = True
            nao_dg3 = []

        if nnz_pw_b:
            fl = [x for x in line.split()]
            if fl[0][0] == '[':
                fl[0] = fl[0][1:]
            elif fl[-1][-1] == ']':
                fl[-1] = fl[-1][:-1]
            nnz_eri_pw += [float(x) for x in fl]

        
        if la_dg_b:
            if 'nnz_eri (PW):' in line:
                la_dg_b  = False
                nnz_pw_b = True
            else:
                fl = [x for x in line.split()]
                if fl[0][0] == '[':
                    if fl[0] == '[':
                        fl = fl[1:]
                    else:
                        fl[0] = fl[0][1:]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                la_dg += [float(x) for x in fl]

        if nnz_dg_b:
            if 'Lambda' in line:
                nnz_dg_b = False
                la_dg_b  = True
            else:
                fl = [x for x in line.split()]
                if fl[0][0] == '[':
                    if fl[0]== '[':
                        fl = fl[1:]
                    else:
                        fl[0] = fl[0][1:]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                nnz_eri_dg += [float(x) for x in fl]

        if la_b:
            if 'Number of AO' in line:
                la_b = False
            else:
                fl = [x for x in line.split()]
                if fl[0][0] == '[':
                    if fl[0] == '[':
                        fl = fl[1:]
                    else:
                        fl[0] = fl[0][1:]
                elif fl[-1][-1] == ']':
                    if fl[-1] == ']':
                        fl = fl[:-1]
                    else:
                        fl[-1] = fl[-1][:-1]
                la += [float(x) for x in fl]

        if nnz_b:
            if 'Lambda' in line:
                nnz_b = False
                la_b  = True
            else:
                fl = [x for x in line.split()]
                if fl[0][0] == '[':
                    fl[0] = fl[0][1:]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                nnz_eri += [float(x) for x in fl]

        if 'nnz_eri (DG):' in line:
            nnz_dg_b = True
        
        if 'nnz_eri:' in line:
            nnz_b = True
  
    nnz_eri     = np.flip(nnz_eri)
    nnz_eri_pw  = np.flip(nnz_eri_pw)
    nnz_eri_dg  = np.flip(nnz_eri_dg)
    nnz_eri_dg1 = np.flip(nnz_eri_dg1)
    nnz_eri_dg2 = np.flip(nnz_eri_dg2)

    la     = np.flip(la)
    la_dg  = np.flip(la_dg)
    la_dg1 = np.flip(la_dg1)
    la_dg2 = np.flip(la_dg2)

    nao_dg1 = np.flip(nao_dg1)
    nao_dg2 = np.flip(nao_dg2)
    nao_dg3 = np.flip(nao_dg3)

    popt_eri, pcov_eri = curve_fit(func1, np.log(np.array(atoms)), 
                                   np.log(np.array(nnz_eri)))
    popt_la, pcov_la = curve_fit(func1, np.log(np.array(atoms[1:])), 
                                 np.log(np.array(la[1:])))
    
    popt_eri_pw, pcov_eri_pw = curve_fit(func1, np.log(np.array(atoms[1:])), 
                                         np.log(np.array(nnz_eri_pw[1:])))
    
    popt_eri_dg, pcov_eri_dg = curve_fit(func1, np.log(np.array(atoms[1:])), 
                                         np.log(np.array(nnz_eri_dg[1:])))
    
    popt_la_dg, pcov_la_dg = curve_fit(func1, np.log(np.array(atoms[1:])), 
                                       np.log(np.array(la_dg[1:])))
    
    popt_eri_dg1, pcov_eri_dg1 = curve_fit(func1, np.log(np.array(atoms[1:])), 
                                           np.log(np.array(nnz_eri_dg1[1:])))
    
    popt_la_dg1, pcov_la_dg1 = curve_fit(func1, np.log(np.array(atoms[1:])), 
                                         np.log(np.array(la_dg1[1:])))
    
    popt_eri_dg2, pcov_eri_dg2 = curve_fit(func1, np.log(np.array(atoms[1:])), 
                                           np.log(np.array(nnz_eri_dg2[1:])))
    
    popt_la_dg2, pcov_la_dg2 = curve_fit(func1, np.log(np.array(atoms[1:])), 
                                         np.log(np.array(la_dg2[1:])))

    #popt_eri[1] = 7.2
    #popt_eri[0] = 2.75

    sp_dg, intc_dg, _, _, _ = stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(nnz_eri_dg[1:])))

    sp_dg1, intc_dg1, _, _, _ = stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(nnz_eri_dg1[1:])))

    sp_dg2, intc_dg2, _, _, _ = stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(nnz_eri_dg2[1:])))

    sp_la_dg, intc_la_dg,_,_,_ = stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(la_dg[1:])))

    sp_la_dg1, intc_la_dg1,_,_,_=stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(la_dg1[1:])))

    sp_la_dg2, intc_la_dg2,_,_,_=stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(la_dg2[1:])))

    

    popt_la_dg[1] = intc_la_dg
    popt_la_dg[0] = sp_la_dg
    
    popt_la_dg1[1] = intc_la_dg1
    popt_la_dg1[0] = sp_la_dg1

    popt_la_dg2[1] = intc_la_dg2
    popt_la_dg2[0] = sp_la_dg2
    

    popt_eri_dg[1] = intc_dg
    popt_eri_dg[0] = sp_dg
    
    popt_eri_dg1[1] = intc_dg1
    popt_eri_dg1[0] = sp_dg1
    
    popt_eri_dg2[1] = intc_dg2
    popt_eri_dg2[0] = sp_dg2

    
    
    arr1[0].loglog(atoms[1:], 
                   np.exp(popt_eri[1])*np.array(atoms[1:])**popt_eri[0], 'k--')
    
    arr1[0].loglog(atoms[1:], 
                   np.exp(popt_eri_pw[1])*np.array(atoms[1:])**popt_eri_pw[0], 
                   'k--')
    
    arr1[0].loglog(atoms[1:], 
                   np.exp(popt_eri_dg[1])*np.array(atoms[1:])**popt_eri_dg[0], 
                   'k--')
    
    arr1[0].loglog(atoms[1:], 
                   np.exp(popt_eri_dg1[1])*np.array(atoms[1:])**popt_eri_dg1[0], 
                   'k--')
    
    arr1[0].loglog(atoms[1:], 
                   np.exp(popt_eri_dg2[1])*np.array(atoms[1:])**popt_eri_dg2[0], 
                   'k--')
    
    arr1[0].loglog(atoms[1:], nnz_eri_pw[1:], color= 'tab:blue', 
                   linestyle = '-.', 
                   label = 'Primitive Basis (${\\alpha}$ = %5.3f)' \
                           % popt_eri_pw[0])
    

    arr1[0].loglog(atoms[1:], nnz_eri[1:], color= 'tab:orange', marker = '*', 
                    label = 'Gaussian (${\\alpha}$ = %5.3f)' % popt_eri[0])
    
    arr1[0].loglog(atoms[1:], nnz_eri_dg2[1:], color = 'tab:green', marker = '^', 
                    label = 'DG $10^{-1}$ (${\\alpha}$ = %5.3f)' \
                            % popt_eri_dg2[0])
    
    arr1[0].loglog(atoms[1:], nnz_eri_dg1[1:], color = 'tab:red', marker = '^', 
                    label = 'DG $10^{-2}$ (${\\alpha}$ = %5.3f)' \
                            % popt_eri_dg1[0])
    
    arr1[0].loglog(atoms[1:], nnz_eri_dg[1:], color = 'tab:purple', marker = '^', 
                    label = 'DG $10^{-3}$ (${\\alpha}$ = %5.3f)' % popt_eri_dg[0])
    
    arr1[0].legend()
    arr1[0].set_xlim(3,32)
    arr1[0].xaxis.set_ticks([4,8,16,32])
    arr1[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    arr2.loglog(atoms[1:],
                   np.exp(popt_eri[1])*np.array(atoms[1:])**popt_eri[0], 'k--')

    arr2.loglog(atoms[1:],
                   np.exp(popt_eri_pw[1])*np.array(atoms[1:])**popt_eri_pw[0],
                   'k--')

    arr2.loglog(atoms[1:],
                   np.exp(popt_eri_dg[1])*np.array(atoms[1:])**popt_eri_dg[0],
                   'k--')

    arr2.loglog(atoms[1:],
                   np.exp(popt_eri_dg1[1])*np.array(atoms[1:])**popt_eri_dg1[0],
                   'k--')
    arr2.loglog(atoms[1:],
                   np.exp(popt_eri_dg2[1])*np.array(atoms[1:])**popt_eri_dg2[0],
                   'k--')

    arr2.loglog(atoms[1:], nnz_eri_pw[1:], color= 'tab:blue',
                   linestyle = '-.',
                   label = 'Primitive Basis (${\\alpha}$ = %5.3f)' \
                           % popt_eri_pw[0])
    
    nao_np   = np.linspace(20, 150, num = 14)
    #nnz_eri_np = nao_np**4 - nao_np**2 
    #arr2.loglog(atoms[1:], nnz_eri_np, color= 'tab:orange', marker = '*',
    #                label = 'Gaussian (${\\alpha}$ = %5.3f)' % popt_eri[0])

    arr2.loglog(atoms[1:], nnz_eri[1:], color= 'tab:orange', marker = '*',
                    label = 'Gaussian (${\\alpha}$ = %5.3f)' % popt_eri[0])

    arr2.loglog(atoms[1:], nnz_eri_dg2[1:], color = 'tab:green', marker = '^',
                    label = 'DG $10^{-1}$ (${\\alpha}$ = %5.3f)' \
                            % popt_eri_dg2[0])

    arr2.loglog(atoms[1:], nnz_eri_dg1[1:], color = 'tab:red', marker = '^',
                    label = 'DG $10^{-2}$ (${\\alpha}$ = %5.3f)' \
                            % popt_eri_dg1[0])

    arr2.loglog(atoms[1:], nnz_eri_dg[1:], color = 'tab:purple', marker = '^',
                    label = 'DG $10^{-3}$ (${\\alpha}$ = %5.3f)' % popt_eri_dg[0])
    arr2.set_xlabel('Number of hydrogens')
    arr2.set_ylabel('Non-zero two-electron integrals')
    arr2.set_ylim(1e3,1e10)
    arr2.set_xlim(3,32)
    arr2.xaxis.set_ticks([4,8,16,32])
    arr2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    arr2.spines['top'].set_visible(False)
    arr2.spines['right'].set_visible(False)
    arr2.legend(frameon=False, loc = 4)


    arr1[1].loglog(atoms[1:], 
            np.exp(popt_la_dg[1])*np.array(atoms[1:])**popt_la_dg[0], 'k--')
    
    arr1[1].loglog(atoms[1:], 
            np.exp(popt_la_dg1[1])*np.array(atoms[1:])**popt_la_dg1[0], 'k--')
    
    arr1[1].loglog(atoms[1:], 
            np.exp(popt_la_dg2[1])*np.array(atoms[1:])**popt_la_dg2[0], 'k--')
    
    arr1[1].loglog(atoms[1:], 
            np.exp(popt_la[1])*np.array(atoms[1:])**popt_la[0], 'k--')
    
    arr1[1].loglog(atoms[1:], la[1:]   , color = 'tab:orange', marker = '*',    
                    label = 'Gaussian (${\\alpha}$ = %5.3f)' % popt_la[0])
    
    arr1[1].loglog(atoms[1:], la_dg2[1:], color = 'tab:green', marker = '^', 
                    label = 'DG $10^{-1}$ (${\\alpha}$ = %5.3f)' % popt_la_dg2[0])
    
    arr1[1].loglog(atoms[1:], la_dg1[1:], color = 'tab:red', marker = '^', 
                    label = 'DG $10^{-2}$ (${\\alpha}$ = %5.3f)' % popt_la_dg1[0])
    
    arr1[1].loglog(atoms[1:], la_dg[1:], color = 'tab:purple', marker = '^', 
                    label = 'DG $10^{-3}$ (${\\alpha}$ = %5.3f)' % popt_la_dg[0])
    arr1[1].set_xlim(3,32)
    arr1[1].xaxis.set_ticks([4,8,16,32])
    arr1[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    arr1[1].legend()


    lambda_PW = np.array([0.1996, 0.3336, 0.4897, 0.6500, 0.8279, 1.0048, 
        1.1978, 1.3870, 1.5916, 1.7906])
    lambda_PW = 1.0e8 * lambda_PW


    arr3.loglog(atoms[1:],
            np.exp(popt_la_dg[1])*np.array(atoms[1:])**popt_la_dg[0], 'k--')

    arr3.loglog(atoms[1:],
            np.exp(popt_la_dg1[1])*np.array(atoms[1:])**popt_la_dg1[0], 'k--')

    arr3.loglog(atoms[1:],
            np.exp(popt_la_dg2[1])*np.array(atoms[1:])**popt_la_dg2[0], 'k--')

    arr3.loglog(atoms[1:],
            np.exp(popt_la[1])*np.array(atoms[1:])**popt_la[0], 'k--')

    
    arr3.loglog(atoms[1:-5], lambda_PW[1:], color = 'tab:blue', ls = '-.',
                    label = 'Gaussian (${\\alpha}$ = %5.3f)' % popt_la[0])

    arr3.loglog(atoms[1:], la[1:]   , color = 'tab:orange', marker = '*',
                    label = 'Gaussian (${\\alpha}$ = %5.3f)' % popt_la[0])

    arr3.loglog(atoms[1:], la_dg2[1:], color = 'tab:green', marker = '^',
                    label = 'DG $10^{-1}$ (${\\alpha}$ = %5.3f)' % popt_la_dg2[0])

    arr3.loglog(atoms[1:], la_dg1[1:], color = 'tab:red', marker = '^',
                    label = 'DG $10^{-2}$ (${\\alpha}$ = %5.3f)' % popt_la_dg1[0])

    arr3.loglog(atoms[1:], la_dg[1:], color = 'tab:purple', marker = '^',
                    label = 'DG $10^{-3}$ (${\\alpha}$ = %5.3f)' % popt_la_dg[0])

    arr3.set_xlabel('Number of hydrogens')
    arr3.set_ylabel('$\lambda$')
    arr3.set_ylim(1e2,4e8)
    arr3.set_xlim(3,32)
    arr3.xaxis.set_ticks([4,8,16,32])
    arr3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    arr3.spines['top'].set_visible(False)
    arr3.spines['right'].set_visible(False)
    arr3.legend(frameon=False, loc = 6)

    arr1[2].plot(atoms, [j/i for i,j in zip(atoms, nao_dg1)], 
            color = 'tab:green', marker = '^', label = 'DG $10^{-1}$')
    
    arr1[2].plot(atoms, [j/i for i,j in zip(atoms, nao_dg2)],
            color = 'tab:red', marker = '^', label = 'DG $10^{-2}$')
    
    arr1[2].plot(atoms, [j/i for i,j in zip(atoms, nao_dg3)],
            color = 'tab:purple', marker = '^', label = 'DG $10^{-3}$')
    arr1[2].set_xlim(3,32)
    arr1[2].xaxis.set_ticks([4,8,16,32])
    arr1[2].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    arr1[2].legend()
    
    arr4.plot(atoms, [j/i for i,j in zip(atoms, nao_dg1)], 
            color = 'tab:green', marker = '^', label = 'DG $10^{-1}$')

    arr4.plot(atoms, [j/i for i,j in zip(atoms, nao_dg2)],
            color = 'tab:red', marker = '^', label = 'DG $10^{-2}$')

    arr4.plot(atoms, [j/i for i,j in zip(atoms, nao_dg3)],
            color = 'tab:purple', marker = '^', label = 'DG $10^{-3}$')
    arr4.set_xlabel('Number of hydrogens')
    arr4.set_ylabel('Mean VdG basis functions per atom')
    arr4.set_ylim(9,20)
    arr4.set_xlim(0,32)
    arr4.xaxis.set_ticks([5,10,15,20,25,30])
    arr4.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    arr4.spines['top'].set_visible(False)
    arr4.spines['right'].set_visible(False)
    arr4.legend(frameon=False, loc = 4)
     
    plt.show()

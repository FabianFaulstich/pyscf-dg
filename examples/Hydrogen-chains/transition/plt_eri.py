import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

import numpy as np
from scipy.optimize import curve_fit

def func(x, a,c):
    return  c*x**a 

def func1(x, a, c):
    return  a*x + c

if __name__ == '__main__':

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    f = open("out_tol1e-3_n.txt", "r")
    g = open("out_tol1e-2_n.txt", "r")
    j = open("out_tol1e-1_n.txt", "r")

    fig1, arr1 = plt.subplots(nrows=1, ncols=3,  figsize=(15,6))
    fig2, arr2 = plt.subplots(nrows=1, ncols=1,  figsize=(7,6))
    fig3, arr3 = plt.subplots(nrows=1, ncols=1,  figsize=(7,6))
    fig4, arr4 = plt.subplots(nrows=1, ncols=1,  figsize=(7,6))
    
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
                if hf[0] == '[':
                    hf = hf[1:]
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
                if hf[0] == '[':
                    hf = hf[1:]
                elif hf[-1][-1] == ']':
                    hf[-1] = hf[-1][:-1]
                nao += [float(x) for x in hf]

        if "Number of AO's:" in line:
            n_ao_b = True
            nao = []

        if la_dg_b:
            fl = [x for x in line.split()]
            if fl[0] == '[':
                fl = fl[1:]
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
                if fl[0] == '[':
                    fl = fl[1:]
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
                if hf[0] == '[':
                    hf = hf[1:]
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
                if fl[0] == '[':
                    fl = fl[1:]
                elif fl[-1] == ']':
                    fl = fl[:-1]
                la_dg1 += [float(x) for x in fl]

        if nnz_dg_b:
            if 'Lambda' in line:
                nnz_dg_b = False
                la_dg_b  = True
            else:
                fl = [x for x in line.split()]
                print(fl)
                if fl[0][0] == '[':
                    if fl[0] == '[':
                        fl = fl[1:]
                    else:
                        fl[0] = fl[0][1:]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                print(fl[0] == '')
                nnz_eri_dg1 += [float(x) for x in fl]

        if 'nnz_eri (DG):' in line:
            nnz_dg_b = True
        
    for cnt, line in enumerate(f):
 
        if n_ao_dg_b:
            if "nnz_eri" in line:
                n_ao_dg_b = False
            else:
                hf = line.split()
                if hf[0] == '[':
                    hf = hf[1:]
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
                if fl[0] == '[':
                    fl = fl[1:]
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
                    fl[0] = fl[0][1:]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                nnz_eri_dg += [float(x) for x in fl]

        if la_b:
            if 'Number of AO' in line:
                la_b = False
            else:
                fl = [x for x in line.split()]
                if fl[0] == '[':
                    fl = fl[1:]
                elif fl[-1] == ']':
                    fl = fl[:-1]
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
   
    popt_eri, pcov_eri = curve_fit(func1, np.log(np.array(atoms[1:])), 
                                   np.log(np.array(nnz_eri[1:])))
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
    arr2.legend()


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
    
    arr1[1].legend()


    arr3.loglog(atoms[1:],
            np.exp(popt_la_dg[1])*np.array(atoms[1:])**popt_la_dg[0], 'k--')

    arr3.loglog(atoms[1:],
            np.exp(popt_la_dg1[1])*np.array(atoms[1:])**popt_la_dg1[0], 'k--')

    arr3.loglog(atoms[1:],
            np.exp(popt_la_dg2[1])*np.array(atoms[1:])**popt_la_dg2[0], 'k--')

    arr3.loglog(atoms[1:],
            np.exp(popt_la[1])*np.array(atoms[1:])**popt_la[0], 'k--')

    arr3.loglog(atoms[1:], la[1:]   , color = 'tab:orange', marker = '*',
                    label = 'Gaussian (${\\alpha}$ = %5.3f)' % popt_la[0])

    arr3.loglog(atoms[1:], la_dg2[1:], color = 'tab:green', marker = '^',
                    label = 'DG $10^{-1}$ (${\\alpha}$ = %5.3f)' % popt_la_dg2[0])

    arr3.loglog(atoms[1:], la_dg1[1:], color = 'tab:red', marker = '^',
                    label = 'DG $10^{-2}$ (${\\alpha}$ = %5.3f)' % popt_la_dg1[0])

    arr3.loglog(atoms[1:], la_dg[1:], color = 'tab:purple', marker = '^',
                    label = 'DG $10^{-3}$ (${\\alpha}$ = %5.3f)' % popt_la_dg[0])

    arr3.legend()

    arr1[2].plot(atoms, [j/i for i,j in zip(atoms, nao_dg1)], 
            color = 'tab:green', marker = '^', label = 'DG $10^{-1}$')
    
    arr1[2].plot(atoms, [j/i for i,j in zip(atoms, nao_dg2)],
            color = 'tab:red', marker = '^', label = 'DG $10^{-2}$')
    
    arr1[2].plot(atoms, [j/i for i,j in zip(atoms, nao_dg3)],
            color = 'tab:purple', marker = '^', label = 'DG $10^{-3}$')
    
    arr1[2].legend()
    
    arr4.plot(atoms, [j/i for i,j in zip(atoms, nao_dg1)], 
            color = 'tab:green', marker = '^', label = 'DG $10^{-1}$')

    arr4.plot(atoms, [j/i for i,j in zip(atoms, nao_dg2)],
            color = 'tab:red', marker = '^', label = 'DG $10^{-2}$')

    arr4.plot(atoms, [j/i for i,j in zip(atoms, nao_dg3)],
            color = 'tab:purple', marker = '^', label = 'DG $10^{-3}$')
    arr4.set_xlabel('Number of hydrogens')
    arr4.set_ylabel('Mean VdG basis functions per atom')
    
    arr4.legend()
     
    plt.show()

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

import numpy as np

if __name__ == '__main__':


    f = open("out_tol-1.txt", "r")
    g = open("out_tol-2_1.txt", "r")
    h = open("out_tol-3_1.txt", "r")

    bonds = [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2]

    ao_b     = False
    mf_b     = False
    mf_b_dg  = False
    mf_b_dg1 = False
    mf_b_dg2 = False
    
    nao_dg1 = []
    nao_dg2 = []
    nao_dg3 = []
    mfe     = []
    mfe_dg1 = []
    mfe_dg2 = []
    mfe_dg3 = []
    
    for line in h:

        if mf_b_dg2:
            hf = line.split()
            if hf[0][0] == '[':
                if hf[0] == '[':
                    hf = hf[1:]
                else:
                    hf[0] = hf[0][1:]
            if hf[-1][-1] == ']':
                if hf[-1] == ']':
                    hf = hf[:-1]
                else:
                    hf[-1] = hf[-1][:-1]
            mfe_dg3 += [float(x) for x in hf]

        if ao_b:
            if 'Hartree--Fock energy:' in line:
                ao_b = False
            else:
                hf = line.split()
                if hf[0][0] == '[':
                    if hf[0] == '[':
                        hf = hf[1:]
                    else:
                        hf[0] = hf[0][1:]
                if hf[-1][-1] == ']':
                    if hf[-1] == ']':
                        hf = hf[:-1]
                    else:
                        hf[-1] = hf[-1][:-1]
                nao_dg3 += [float(x) for x in hf]

        if "Number of AO's (VdG):" in line:
             ao_b = True

        if "Hartree--Fock energy (VdG):" in line:
            mf_b_dg2 = True


    for line in g:

        if mf_b_dg1:
            hf = line.split()
            if hf[0][0] == '[':
                if hf[0] == '[':
                    hf = hf[1:]
                else:
                    hf[0] = hf[0][1:]
            if hf[-1][-1] == ']':
                if hf[-1] == ']':
                    hf = hf[:-1]
                else:
                    hf[-1] = hf[-1][:-1]
            mfe_dg2 += [float(x) for x in hf]

        if ao_b:
            if 'Hartree--Fock energy:' in line:
                ao_b = False
            else:
                hf = line.split()
                if hf[0][0] == '[':
                    if hf[0] == '[':
                        hf = hf[1:]
                    else:
                        hf[0] = hf[0][1:]
                if hf[-1][-1] == ']':
                    if hf[-1] == ']':
                        hf = hf[:-1]
                    else:
                        hf[-1] = hf[-1][:-1]
                nao_dg2 += [float(x) for x in hf]

        if "Number of AO's (VdG):" in line:
             ao_b = True

        if "Hartree--Fock energy (VdG):" in line:
            mf_b_dg1 = True

    for line in f:
        
        if mf_b_dg:
            hf = line.split()
            if hf[0][0] == '[':
                if hf[0] == '[':
                    hf = hf[1:]
                else:
                    hf[0] = hf[0][1:]
            if hf[-1][-1] == ']':
                if hf[-1] == ']':
                    hf = hf[:-1]
                else:
                    hf[-1] = hf[-1][:-1]
            mfe_dg1 += [float(x) for x in hf]

        if mf_b:
            if 'Hartree--Fock energy (VdG):' in line:
                mf_b = False
                mf_b_dg = True
            else:
                hf = line.split()
                if hf[0][0] == '[':
                    if hf[0] == '[':
                        hf = hf[1:]
                    else:
                        hf[0] = hf[0][1:]
                if hf[-1][-1] == ']':
                    if hf[-1] == ']':
                        hf = hf[:-1]
                    else:
                        hf[-1] = hf[-1][:-1]
                mfe += [float(x) for x in hf]

        if ao_b:
            if 'Hartree--Fock energy:' in line:
                ao_b = False
                mf_b = True
            else:
                hf = line.split()
                if hf[0][0] == '[':
                    if hf[0] == '[':
                        hf = hf[1:]
                    else:
                        hf[0] = hf[0][1:]
                if hf[-1][-1] == ']':
                    if hf[-1] == ']':
                        hf = hf[:-1]
                    else:
                        hf[-1] = hf[-1][:-1]
                nao_dg1 += [float(x) for x in hf]
        
        if "Number of AO's (VdG):" in line:
             ao_b = True


    av_dg1 = int(np.ceil(np.sum(np.array(nao_dg1))/len(nao_dg1)/16))
    av_dg2 = int(np.ceil(np.sum(np.array(nao_dg2))/len(nao_dg2)/16))
    av_dg3 = int(np.ceil(np.sum(np.array(nao_dg3))/len(nao_dg3)/16))
   
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams.update({'font.size': 12})

    fig, arr = plt.subplots(figsize=(6,4))

    #plt.figure(figsize=(6,4))
    arr.plot(bonds[:-1], mfe[:-1], color= 'tab:orange', marker = '*', 
            label = 'Gaussian (cc-pVDZ)')
    arr.plot(bonds[:-1], mfe_dg1[:-1], color = 'tab:green', marker = '^',
            label = 'DG $10^{-1}$, $\langle n_k\\rangle$ =' + str(av_dg1))
    arr.plot(bonds[:-1], mfe_dg2[:], color = 'tab:red', marker = '^',
            label = 'DG $10^{-2}$, $\langle n_k\\rangle$ =' + str(av_dg2))
    arr.plot(bonds[:-1], mfe_dg3[:], color = 'tab:purple', marker = '^', 
            label = 'DG $10^{-3}$, $\langle n_k\\rangle$ =' + str(av_dg3) )
    arr.spines['top'].set_visible(False)
    arr.spines['right'].set_visible(False)
    arr.legend(loc = 'upper center', frameon = False)
    arr.set_xlabel("Internuclear distance (a.u)")
    arr.set_ylabel("Energy (a.u)")
    arr.set_xlim((1.1, 2.9)) 
    fig.tight_layout()
    plt.show()


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

import numpy as np

if __name__ == '__main__':

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    mf_b  = False
    rot_b = False

    f   = open("out_321g.txt", "r")

    fig, arr = plt.subplots(nrows=4, ncols=2,  figsize=(20,8))

    for line in f:

        if 'AO basis' in line:
            fl = line.split()
            BS = fl[1][6:]
        
        if 'SVD ' in line:
            fl = line.split()
            svd_tol = fl[1][11:]

        if rot_b:
            if 'Mean-field' in line:
                #print(emf)
                rot_b = False
            else: 
                hf = line.split()
                if hf[-1][-1] == ']':
                    hf[-1] = hf[-1][:-1]
                rot += [float(x) for x in hf]

        if 'Rotation angle' in line:
            hf = line.split()[2:]
            hf[0] = hf[0][1:]
            rot = [float(x) for x in hf]
            rot_b = True

        if mf_b:
            if 'Mean-field energy (DG)' in line:
                mf_b = False
            else:
                hf = line.split()
                print(hf)

        if 'Mean-field energy:' in line:
            hf = line.split()[2:]
            if hf[0][0] == '[':
                hf[0] = hf[0][1:]
            elif hf[0] == '[':
                hf = hf[1:]
            print(hf)
            emf = [float(x) for x in hf]
            mf_b = True


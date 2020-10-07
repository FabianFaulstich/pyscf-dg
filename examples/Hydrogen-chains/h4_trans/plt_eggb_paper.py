import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

import numpy as np

if __name__ == '__main__':

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    bs       = False
    shift_b  = False
    mfe_b    = False
    mfe_dg_b = False
    mpe_b    = False
    mpe_dg_b = False
    plt_b    = False

    f   = open("out_egg_box_fine_small1.txt", "r")
    
    fig1, arr1 = plt.subplots(nrows=1, ncols=3,  figsize=(12,3))
    fig2, arr2 = plt.subplots(nrows=1, ncols=3,  figsize=(12,3))

    for cnt, line in enumerate(f):
        
        if bs:
            BS = line
            bs = False
        
        if 'AO basis' in line:
            bs = True

        if shift_b:
            if 'Mean-field energy:' in line:
                shift_b = False
            else:
                fl = [x for x in line.split()]
                if fl[0][-1] == ']':
                    fl[0]= fl[0][:-1]
                elif fl[-1] == ']':
                    fl = fl[:-1]
                shift += [float(x) for x in fl]

        if 'Shift along the X-coordinate:' in line:
            fl = [x for x in line.split()]
            fl = fl[4:]
            fl[0] = fl[0][-2:]
            shift = [float(x) for x in fl]
            shift_b = True

        if mfe_b: 
            if 'MP2 corr. energy :' in line:
                mfe_b = False
            else:
                fl = [x for x in line.split()]
                if fl[0][-1] == ']':
                    fl[0]= fl[0][:-1]
                elif fl[-1] == ']':
                    fl = fl[:-1]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                mfe += [float(x) for x in fl]
        
        if 'Mean-field energy:' in line:
            fl = [x for x in line.split()]
            fl = fl[2:]
            fl[0] = fl[0][1:]
            mfe = [float(x) for x in fl]
            mfe_b = True

        if mpe_b:
            if 'Mean-field energy (DG):' in line:
                mpe_b = False
            else:
                fl = [x for x in line.split()]
                if fl[0][-1] == ']':
                    fl[0]= fl[0][:-1]
                elif fl[-1] == ']':
                    fl = fl[:-1]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                mpe += [float(x) for x in fl]
    
        if 'MP2 corr. energy :' in line:
            fl = [x for x in line.split()]
            fl = fl[4:]
            fl[0] = fl[0][1:]
            mpe = [float(x) for x in fl]
            mpe_b = True

        if mfe_dg_b:
            if 'MP2 corr. energy (DG) :' in line:
                mfe_dg_b = False
            else:
                fl = [x for x in line.split()]
                if fl[0][-1] == ']':
                    fl[0]= fl[0][:-1]
                elif fl[-1] == ']':
                    fl = fl[:-1]
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                mfe_dg += [float(x) for x in fl]

        if 'Mean-field energy (DG):' in line:
            fl    = [x for x in line.split()]
            fl    = fl[3:]
            fl[0] = fl[0][1:]
            mfe_dg = [float(x) for x in fl]
            mfe_dg_b = True

        if mpe_dg_b:
            if 'Rotation angle' in line:
                mpe_dg_b = False
            else:
                fl = [x for x in line.split()]
                if fl[0][-1] == ']':
                    fl[0]= fl[0][:-1]
                    plt_b = True
                elif fl[-1] == ']':
                    fl = fl[:-1]
                    plt_b = True
                elif fl[-1][-1] == ']':
                    fl[-1] = fl[-1][:-1]
                    plt_b = True
                mpe_dg += [float(x) for x in fl]

        if 'MP2 corr. energy (DG) :' in line:
            fl = [x for x in line.split()]
            fl = fl[5:]
            fl[0] = fl[0][1:]
            mpe_dg = [float(x) for x in fl]
            mpe_dg_b = True

        if plt_b:
            plt_b = False
            if 'ccpvdz' in BS and 'aug' not in BS:

                tot_bi  = np.array([a + b for a, b in zip(mfe, mpe)])
                tot_dg  = np.array([a + b for a, b in zip(mfe_dg, mpe_dg)])

                mfe     = np.array(mfe)
                mfe_dg  = np.array(mfe_dg)

                mpe     = np.array(mpe)
                mpe_dg  = np.array(mpe_dg)

                arr1[0].plot(shift[1:80], tot_dg[1:80] - np.mean(tot_dg[1:80]),
                        color = 'r', marker = '.', label = 'DG-cc-pVSD')
                arr1[0].plot(shift[1:80], tot_bi[1:80] - np.mean(tot_bi[1:80]),
                        color = 'b', marker = '.', label = 'cc-pVDZ')
                #arr1[0].legend()
                arr1[0].set_ylim(-16e-6,13e-6)
                arr1[0].set_title("Total Energy ")

                arr1[1].plot(shift[1:80], mfe[1:80] - np.mean(mfe[1:80]),
                        color = 'b', marker = '.', label = 'PySCF')
                arr1[1].plot(shift[1:80], mfe_dg[1:80] - np.mean(mfe_dg[1:80]),
                        color = 'r', marker = '.', label = 'cc-pVDZ')
                arr1[1].set_yticklabels([])
                #arr1[0,1].legend()
                arr1[1].set_ylim(-16e-6,13e-6)
                arr1[1].set_title("Mean Field Energy ")
                
                arr1[2].plot(shift[1:80], mpe[1:80] - np.mean(mpe[1:80]),
                        color = 'b', marker = '.', label = 'PySCF')
                arr1[2].plot(shift[1:80], mpe_dg[1:80] - np.mean(mpe_dg[1:80]),
                        color = 'r', marker = '.', label = 'cc-pVDZ')
                arr1[2].set_yticklabels([])
                arr1[2].legend(loc = 'lower right')
                arr1[2].set_ylim(-16e-6,13e-6)
                arr1[2].set_title("MP2 Energy ")

                       
            if '631g' in BS:

                tot_bi  = np.array([a + b for a, b in zip(mfe, mpe)])
                tot_dg  = np.array([a + b for a, b in zip(mfe_dg, mpe_dg)])

                mfe     = np.array(mfe)
                mfe_dg  = np.array(mfe_dg)

                mpe     = np.array(mpe)
                mpe_dg  = np.array(mpe_dg)

                arr2[0].plot(shift[1:80], tot_dg[1:80] - np.mean(tot_dg[1:80]),
                        color = 'r', marker = '.', label = 'DG-6-31G')
                arr2[0].plot(shift[1:80], tot_bi[1:80] - np.mean(tot_bi[1:80]),
                        color = 'b', marker = '.', label = '6-31G')
                arr2[0].set_ylim(-42e-6,38e-6)
                y_labels = arr2.get_yticks()
                arr2.set_yticklabels(['1e%i' % np.round(np.log(y)/np.log(10)) for y in y_labels])
                #arr1[0,1].set_yticklabels([])
                #arr2[0].legend(loc = 'upper right')
                arr2[0].set_title("Total Energy ")

                arr2[1].plot(shift[1:80], mfe[1:80] - np.mean(mfe[1:80]),
                        color = 'b', marker = '.', label = 'PySCF')
                arr2[1].plot(shift[1:80], mfe_dg[1:80] - np.mean(mfe_dg[1:80]),
                        color = 'r', marker = '.', label = '6-31G')
                #arr1[0,1].legend()
                arr2[1].set_ylim(-42e-6,38e-6)
                arr2[1].set_yticklabels([])
                arr2[1].set_title("Mean Field Energy ")

                arr2[2].plot(shift[1:80], mpe[1:80] - np.mean(mpe[1:80]),
                        color = 'b', marker = '.', label = 'PySCF')
                arr2[2].plot(shift[1:80], mpe_dg[1:80] - np.mean(mpe_dg[1:80]),
                        color = 'r', marker = '.', label = '6-31G')
                arr2[2].legend(loc = 'lower right')
                arr2[2].set_ylim(-42e-6,38e-6)
                arr2[2].set_yticklabels([])
                arr2[2].set_title("MP2 Energy ")


    plt.show()


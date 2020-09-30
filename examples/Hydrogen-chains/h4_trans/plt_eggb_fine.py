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

    f   = open("out_egg_box_fine.txt", "r")

    fig4, arr4 = plt.subplots(nrows=3, ncols=3,  figsize=(20,8))
    fig6, arr6 = plt.subplots(nrows=3, ncols=3,  figsize=(20,8))

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
            print(fl)
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

                arr4[0,0].plot(shift, tot_bi,# - np.mean(tot_bi), 
                        color = 'b', marker = '^', label = 'PySCF')
                arr4[1,0].plot(shift, tot_dg,# - np.mean(tot_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr4[2,0].plot(shift, tot_dg - np.mean(tot_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr4[2,0].plot(shift, tot_bi - np.mean(tot_bi),
                        color = 'b', marker = '^', label = 'PySCF')
                arr4[2,0].legend()

                arr4[0,0].set_title("Total Energy " + BS)
                arr4[1,0].legend()
                arr4[0,0].legend()

                arr4[0,1].plot(shift, mfe,# - np.mean(mfe),
                        color = 'b', marker = '^', label = 'PySCF')
                arr4[1,1].plot(shift, mfe_dg,# - np.mean(mfe_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr4[2,1].plot(shift, mfe - np.mean(mfe),
                        color = 'b', marker = '^', label = 'PySCF')
                arr4[2,1].plot(shift, mfe_dg - np.mean(mfe_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr4[2,1].legend()
                arr4[0,1].set_title("Mean Field Energy " + BS)
                arr4[1,1].legend()
                arr4[0,1].legend()

                arr4[0,2].plot(shift, mpe,
                        color = 'b', marker = '^', label = 'PySCF')
                arr4[1,2].plot(shift, mpe_dg,
                        color = 'r', marker = 'v', label = 'DG')
                arr4[2,2].plot(shift, mpe - np.mean(mpe),
                        color = 'b', marker = '^', label = 'PySCF')
                arr4[2,2].plot(shift, mpe_dg - np.mean(mpe_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr4[2,2].legend()
                arr4[0,2].set_title("MP2 Energy " + BS)
                arr4[1,2].legend()
                arr4[0,2].legend()
            
                       
            if '631g' in BS:

                tot_bi  = np.array([a + b for a, b in zip(mfe, mpe)])
                tot_dg  = np.array([a + b for a, b in zip(mfe_dg, mpe_dg)])

                mfe     = np.array(mfe)
                mfe_dg  = np.array(mfe_dg)

                mpe     = np.array(mpe)
                mpe_dg  = np.array(mpe_dg)

                arr6[0,0].plot(shift, tot_bi,# - np.mean(tot_bi), 
                        color = 'b', marker = '^', label = 'PySCF')
                arr6[1,0].plot(shift, tot_dg,# - np.mean(tot_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr6[2,0].plot(shift, tot_dg - np.mean(tot_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr6[2,0].plot(shift, tot_bi - np.mean(tot_bi),
                        color = 'b', marker = '^', label = 'PySCF')
                arr6[2,0].legend()
                arr6[0,0].set_title("Total Energy " + BS)
                arr6[1,0].legend()
                arr6[0,0].legend()

                arr6[0,1].plot(shift, mfe,# - np.mean(mfe),
                        color = 'b', marker = '^', label = 'PySCF')
                arr6[1,1].plot(shift, mfe_dg,# - np.mean(mfe_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr6[2,1].plot(shift, mfe - np.mean(mfe),
                        color = 'b', marker = '^', label = 'PySCF')
                arr6[2,1].plot(shift, mfe_dg - np.mean(mfe_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr6[2,1].legend()
                arr6[0,1].set_title("Mean Field Energy " + BS)
                arr6[1,1].legend()
                arr6[0,1].legend()

                arr6[0,2].plot(shift, mpe,
                        color = 'b', marker = '^', label = 'PySCF')
                arr6[1,2].plot(shift, mpe_dg,
                        color = 'r', marker = 'v', label = 'DG')
                arr6[2,2].plot(shift, mpe - np.mean(mpe),
                        color = 'b', marker = '^', label = 'PySCF')
                arr6[2,2].plot(shift, mpe_dg - np.mean(mpe_dg),
                        color = 'r', marker = 'v', label = 'DG')
                arr6[2,2].legend()

                arr6[0,2].set_title("MP2 Energy " + BS)
                arr6[1,2].legend()
                arr6[0,2].legend()

           
    plt.show()
























import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

import numpy as np



if __name__ == '__main__':

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams.update({'font.size': 15})

    bs       = False
    shift_b  = False
    mfe_b    = False
    mfe_dg_b = False
    mpe_b    = False
    mpe_dg_b = False
    plt_b    = False

    f   = open("out_egg_box_fine_small1.txt", "r")
    
    fig11, arr11 = plt.subplots(nrows=1, ncols=1,  figsize=(4,3))
    fig12, arr12 = plt.subplots(nrows=1, ncols=1,  figsize=(4,3))
    fig13, arr13 = plt.subplots(nrows=1, ncols=1,  figsize=(4,3))

    fig21, arr21 = plt.subplots(nrows=1, ncols=1,  figsize=(4,3))
    fig22, arr22 = plt.subplots(nrows=1, ncols=1,  figsize=(4,3))
    fig23, arr23 = plt.subplots(nrows=1, ncols=1,  figsize=(4,3))


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

                arr11.plot(shift[1:80], tot_dg[1:80] - np.mean(tot_dg[1:80]),
                        color = 'r', marker = '.', label = 'DG-cc-pVSD')
                arr11.plot(shift[1:80], tot_bi[1:80] - np.mean(tot_bi[1:80]),
                        color = 'b', marker = '.', label = 'cc-pVDZ')
                arr11.set_ylim(-30e-6,23e-6)
                arr11.ticklabel_format(axis = 'y',
                                       style = 'sci',
                                       scilimits = (5,5))
                arr11.spines['right'].set_visible(False)
                arr11.spines['top'].set_visible(False)
                fig11.tight_layout()                



                arr12.plot(shift[1:80], mfe[1:80] - np.mean(mfe[1:80]),
                        color = 'b', marker = '.', label = 'PySCF')
                arr12.plot(shift[1:80], mfe_dg[1:80] - np.mean(mfe_dg[1:80]),
                        color = 'r', marker = '.', label = 'cc-pVDZ')
                arr12.set_ylim(-20e-6,23e-6)
                arr12.ticklabel_format(axis = 'y',
                                       style = 'sci',
                                       scilimits = (5,5))
                arr12.spines['right'].set_visible(False)
                arr12.spines['top'].set_visible(False)
                fig12.tight_layout()


                arr13.plot(shift[1:80], mpe[1:80] - np.mean(mpe[1:80]),
                        color = 'b', marker = '.', label = 'cc-pVDZ')
                arr13.plot(shift[1:80], mpe_dg[1:80] - np.mean(mpe_dg[1:80]),
                        color = 'r', marker = '.', label = 'DG-V-cc-pVDZ')
                arr13.legend(frameon = False, loc = 'lower right')
                arr13.set_ylim(-30e-6,23e-6)
                arr13.ticklabel_format(axis = 'y',
                                       style = 'sci',
                                       scilimits = (5,5))
                arr13.spines['right'].set_visible(False)
                arr13.spines['top'].set_visible(False)
                fig13.tight_layout() 

                       
            if '631g' in BS:

                tot_bi  = np.array([a + b for a, b in zip(mfe, mpe)])
                tot_dg  = np.array([a + b for a, b in zip(mfe_dg, mpe_dg)])

                mfe     = np.array(mfe)
                mfe_dg  = np.array(mfe_dg)

                mpe     = np.array(mpe)
                mpe_dg  = np.array(mpe_dg)

                arr21.plot(shift[1:80], tot_dg[1:80] - np.mean(tot_dg[1:80]),
                        color = 'r', marker = '.', label = 'DG-V-6-31G')
                arr21.plot(shift[1:80], tot_bi[1:80] - np.mean(tot_bi[1:80]),
                        color = 'b', marker = '.', label = '6-31G')
                arr21.set_ylim(-42e-6,38e-6)
                arr21.ticklabel_format(axis = 'y', 
                                       style = 'sci', 
                                       scilimits = (5,5))
                arr21.spines['right'].set_visible(False)
                arr21.spines['top'].set_visible(False)
                fig21.tight_layout()


                arr22.plot(shift[1:80], mfe[1:80] - np.mean(mfe[1:80]),
                        color = 'b', marker = '.', label = 'PySCF')
                arr22.plot(shift[1:80], mfe_dg[1:80] - np.mean(mfe_dg[1:80]),
                        color = 'r', marker = '.', label = '6-31G')
                arr22.set_ylim(-42e-6,38e-6)
                arr22.ticklabel_format(axis = 'y', 
                                       style = 'sci',
                                       scilimits = (5,5))
                arr22.spines['right'].set_visible(False)
                arr22.spines['top'].set_visible(False)
                fig22.tight_layout()

                arr23.plot(shift[1:80], mpe[1:80] - np.mean(mpe[1:80]),
                        color = 'b', marker = '.', label = '6-31G')
                arr23.plot(shift[1:80], mpe_dg[1:80] - np.mean(mpe_dg[1:80]),
                        color = 'r', marker = '.', label = 'DG-V-6-31G')
                arr23.legend(frameon = False, loc = 'lower right')
                arr23.set_ylim(-42e-6,38e-6)
                arr23.ticklabel_format(axis = 'y', 
                                       style = 'sci',
                                       scilimits = (5,5))
                arr23.spines['right'].set_visible(False)
                arr23.spines['top'].set_visible(False)
                fig23.tight_layout()
                

    arr11.set_xlabel('Displacement (a.u.)')
    arr11.set_ylabel('Energy (a.u.)')
    fig11.tight_layout()
    fig11.subplots_adjust(top=0.92, bottom = .19, left = .17, right = .98)

    arr12.set_xlabel('Displacement (a.u.)')
    arr12.set_ylabel('Energy (a.u.)')
    fig12.tight_layout()
    fig12.subplots_adjust(top=0.92, bottom = .19, left = .17, right = .98)

    arr13.set_xlabel('Displacement (a.u.)')
    arr13.set_ylabel('Energy (a.u.)')
    fig13.tight_layout()
    fig13.subplots_adjust(top=0.92, bottom = .19, left = .17, right = .98)
    
    arr21.set_xlabel('Displacement (a.u.)')
    arr21.set_ylabel('Energy (a.u.)')
    fig21.tight_layout()
    fig21.subplots_adjust(top=0.92, bottom = .19, left = .17, right = .98)

    arr22.set_xlabel('Displacement (a.u.)')
    arr22.set_ylabel('Energy (a.u.)')
    fig22.tight_layout()
    fig22.subplots_adjust(top=0.92, bottom = .19, left = .17, right = .98)

    arr23.set_xlabel('Displacement (a.u.)')
    arr23.set_ylabel('Energy (a.u.)')
    fig23.tight_layout()
    fig23.subplots_adjust(top=0.92, bottom = .19, left = .17, right = .98)
    plt.show()


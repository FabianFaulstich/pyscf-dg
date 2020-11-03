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
    plt.rcParams.update({'font.size': 12.3})


    atoms = np.linspace(2,30, num=15)
    # No Loewdin
    #SVD tol: 0.1
    nao = np.array([ 
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
        ])
    nao_dg1 = np.array([ 
        20, 52, 82, 112, 145, 176, 207, 240, 271, 303, 333, 363, 395, 426, 456
        ])
    nnz_eri = np.array([ 
        2072.0, 36568.0, 190812.0, 606584.0, 1464260.0, 3006184.0, 5500492.0, 
        9089464.0, 14458488.0, 21809512.0, 30924740.0, 42621152.0, 58743516.0,
        78256624.0, 101653352.0
        ])
    nnz_eri_dg1 = np.array([ 
        17728.0, 187828.0, 558756.0, 1065762.0, 1935029.0, 2699892.0, 
        3670381.0, 4463652.0, 5752113.0, 6917989.0, 8015497.0, 9059857.0, 
        10663325.0, 12016172.0, 13437200.0
        ])
    la = np.array([ 
        92.32110876876519, 561.2719842403932, 1703.3512404456728, 
        3473.6921748406676, 6164.799004413331, 9867.013811970475, 
        14778.051637958226, 21236.929873596295, 29486.319175426765, 
        38761.60729460747, 50306.64606375845, 64560.65344206327, 
        80734.68255504587, 98974.97451403661, 118243.93440955966
        ])
    la_dg1 = np.array([ 
        447.61039870970313, 2178.9141761002006, 4634.649872535632, 
        7677.09162760667, 12137.436000549105, 17668.064483130387, 
        24411.031704717323, 32447.36763005875, 42552.853608808306, 
        54425.39637000106, 68478.47682714764, 83763.3073322243,
        102624.56125899105, 123430.69848646497, 146429.26954285614
        ])
    nnz_eri_pw = np.array([
        5.06250000e+10, 9.76562500e+10, 1.60000000e+11, 2.37656250e+11,
        3.30625000e+11, 4.55625000e+11, 5.81406250e+11, 7.22500000e+11,
        8.78906250e+11, 1.05062500e+12, 1.26562500e+12, 1.47015625e+12,
        1.69000000e+12, 1.92515625e+12, 2.17562500e+12
        ])
    la_pw = np.array([
        4.67378516e+08, 9.12659618e+08, 1.51346525e+09, 2.27525511e+09,
        3.20333728e+09, 4.47475500e+09, 5.77788067e+09, 7.26473127e+09,
        8.94109555e+09, 1.08126032e+10, 1.31979693e+10, 1.55074439e+10,
        1.80305159e+10, 2.07733024e+10, 2.37417548e+10
        ])  

    nnz_eri_mo = np.array([
        128.0, 2048.0, 7688.0, 20432.0, 47844.0, 94472.0, 173448.0,
        289892.0, 461012.0, 697172.0, 1018532.0, 1432152.0, 1969548.0,
        2635784.0, 3470712.0
        ])

    la_mo = np.array([
        11.38982712250952, 68.0747562442399, 162.2674183439647,
        331.89678701065606, 598.372541859568, 985.530002714339,
        1537.182926150445, 2291.9687480974294, 3080.363490517499,
        4346.273634203047, 5841.592404884032, 7375.0019696752315,
        9403.916593023128, 11598.660434016756, 14458.453930702959
        ])


    #SVD tol: 0.01
    nao_dg2 = np.array([ 
        20, 64, 106, 147, 190, 231, 273, 316, 
        359, 403, 448, 489, 535, 577, 618
        ])
    nnz_eri_dg2 = np.array([
        17728.0, 443728.0, 1643880.0, 3338971.0, 5879048.0, 8383165.0, 
        11400321.0, 14337240.0, 18492749.0, 22639325.0, 26941916.0, 
        31027141.0, 36664561.0, 41350557.0, 46459372.0
        ])
    la_dg2 = np.array([
        447.61039870970313, 3821.1438289020944, 9600.818202229251, 
        16108.978005080675, 25393.357217214667, 36418.97842597099, 
        49978.980239718905, 65696.96858694851, 86024.68450912315, 
        109842.00526252379, 138988.22298621715, 169242.87947675106,
        207564.66803064468, 247737.39651429703, 292424.7073923659
        ])

    #SVD tol: 0.001
    nao_dg3 = np.array([
        20, 72, 124, 176, 226, 276, 326, 376, 426, 476, 526, 576, 626
        ])
    nnz_eri_dg3 = np.array([
        17728.0, 726260.0, 3176336.0, 6862330.0, 11412216.0, 16075348.0, 
        21442808.0, 26387648.0, 32973128.0, 40011680.0, 46711348.0, 
        53335832.0, 61450636.0
        ])
    la_dg3 = np.array([ 
        447.61039870970313, 5357.520650437367, 14655.152142501165, 
        25090.514528294767, 38354.78907093906, 54864.66149950643, 
        74660.26116179029, 96842.71169227763, 125170.66381161746, 
        158188.76704349596, 197730.87988801047, 240809.44012833497, 
        290909.71809843864
        ])


    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig,      arr      = plt.subplots(nrows=1, ncols=1,  figsize=(5,5))
    fig_nl_1, arr_nl_1 = plt.subplots(nrows=1, ncols=1,  figsize=(6,5))
    fig_nl_2, arr_nl_2 = plt.subplots(nrows=1, ncols=1,  figsize=(6,5))
    fig.subplots_adjust(top=.95)

    

    sp_pw, intc_pw, _, _, _ = stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(nnz_eri_pw[1:])))
    
    sp_mo, intc_mo, _, _, _ = stats.linregress(np.log(np.array(atoms[:])),
            np.log(np.array(nnz_eri_mo[:])))

    sp, intc, _, _, _ = stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(nnz_eri[1:])))

    sp_dg1,intc_dg1,_,_,_ = stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(nnz_eri_dg1[1:])))

    sp_dg2,intc_dg2,_,_,_ = stats.linregress(np.log(np.array(atoms[1:])),
            np.log(np.array(nnz_eri_dg2[1:])))
    
    sp_dg3,intc_dg3,_,_,_ = stats.linregress(np.log(np.array(atoms[1:13])),
            np.log(np.array(nnz_eri_dg3[1:])))

    
    sp_la_pw, intc_la_pw, _, _, _= stats.linregress(
            np.log(np.array(atoms[1:])),
            np.log(np.array(la_pw[1:]))
            )
    sp_la_mo, intc_la_mo, _, _, _= stats.linregress(
            np.log(np.array(atoms[:])),
            np.log(np.array(la_mo[:]))
            )
    
    #sp_la_pw   = 1.12
    #intc_la_pw = 19
    sp_la, intc_la, _, _, _= stats.linregress(
            np.log(np.array(atoms[1:])),
            np.log(np.array(la[1:]))
            )

    sp_la_dg1, intc_la_dg1, _, _, _= stats.linregress(
            np.log(np.array(atoms[1:])),
            np.log(np.array(la_dg1[1:]))
            )

    sp_la_dg2, intc_la_dg2, _, _, _= stats.linregress(
            np.log(np.array(atoms[1:])),
            np.log(np.array(la_dg2[1:]))
            )

    sp_la_dg3, intc_la_dg3, _, _, _= stats.linregress(
            np.log(np.array(atoms[1:13])),
            np.log(np.array(la_dg3[1:])))

  
    
    arr_nl_1.loglog(atoms[1:],
                   np.exp(intc_pw)*np.array(atoms[1:])**sp_pw, 'k--')

    arr_nl_1.loglog(atoms[1:],
                   np.exp(intc_mo)*np.array(atoms[1:])**sp_mo, 'k--')

    arr_nl_1.loglog(atoms[1:],
                   np.exp(intc)*np.array(atoms[1:])**sp, 'k--')

    arr_nl_1.loglog(atoms[1:],
                   np.exp(intc_dg1)*np.array(atoms[1:])**sp_dg1,
                   'k--')

    arr_nl_1.loglog(atoms[1:],
                   np.exp(intc_dg2)*np.array(atoms[1:])**sp_dg2,
                   'k--')

    arr_nl_1.loglog(atoms[1:],
                   np.exp(intc_dg3)*np.array(atoms[1:])**sp_dg3,
                   'k--')


    arr_nl_1.loglog(
            atoms[1:],
            nnz_eri_mo[1:],
            color='tab:cyan',
            marker = '.',
            label = "MO (${\\alpha}$ = %5.3f)" % sp_mo
            )

    arr_nl_1.loglog(
            atoms[1:], 
            nnz_eri_pw[1:],
            color= 'tab:blue', 
            ls = '-.',
            label = 'PW Basis (${\\alpha}$ = %5.3f)' % sp_pw
            )


    arr_nl_1.loglog(
            atoms[1:], 
            nnz_eri[1:], 
            color= 'tab:orange', 
            marker = '*',
            label = 'Gaussian (${\\alpha}$ = %5.3f)' % sp
            )

    arr_nl_1.loglog(
            atoms[1:], 
            nnz_eri_dg1[1:], color = 'tab:green', 
            marker = '^',
            label = 'DG $10^{-1}$ (${\\alpha}$ = %5.3f)' % sp_dg1
            )

    arr_nl_1.loglog(
            atoms[1:], 
            nnz_eri_dg2[1:], 
            color = 'tab:red', 
            marker = '^',
            label = 'DG $10^{-2}$ (${\\alpha}$ = %5.3f)'% sp_dg2
            )

    arr_nl_1.loglog(
            atoms[1:13], 
            nnz_eri_dg3[1:], 
            color = 'tab:purple', 
            marker = '^',
            label = 'DG $10^{-3}$ (${\\alpha}$ = %5.3f)' % sp_dg3
            )

    arr_nl_1.set_xlabel('Number of hydrogens')
    arr_nl_1.set_ylabel('Non-zero two-electron integrals')
    #arr_nl_1.set_ylim(1e3,1e10)
    arr_nl_1.set_xlim(3,32)
    arr_nl_1.xaxis.set_ticks([4,8,16,32])
    arr_nl_1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    arr_nl_1.spines['top'].set_visible(False)
    arr_nl_1.spines['right'].set_visible(False)
    #arr_nl_1.legend(frameon=False,
    #                loc ='center left',
    #                bbox_to_anchor=(0, 0.36, 0.5, 0.5))
    arr_nl_1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=2,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )
    fig_nl_1.tight_layout()
    fig_nl_1.subplots_adjust(top=0.79, bottom = .1)
  


    # Lambda value
    arr_nl_2.loglog(atoms[1:],
                   np.exp(intc_la_mo)*np.array(atoms[1:])**sp_la_mo, 'k--')
    
    arr_nl_2.loglog(atoms[1:],
                   np.exp(intc_la_pw)*np.array(atoms[1:])**sp_la_pw, 'k--')

    arr_nl_2.loglog(atoms[1:],
                   np.exp(intc_la)*np.array(atoms[1:])**sp_la, 'k--')

    arr_nl_2.loglog(atoms[1:],
                   np.exp(intc_la_dg1)*np.array(atoms[1:])**sp_la_dg1,
                   'k--')

    arr_nl_2.loglog(atoms[1:],
                   np.exp(intc_la_dg2)*np.array(atoms[1:])**sp_la_dg2,
                   'k--')

    arr_nl_2.loglog(atoms[1:],
                   np.exp(intc_la_dg3)*np.array(atoms[1:])**sp_la_dg3,
                   'k--')

    arr_nl_2.loglog(
            atoms[1:],
            la_mo[1:],
            color= 'tab:cyan',
            marker = '.',
            label = "MO (${\\alpha}$ = %5.3f)" % sp_la_mo
            )

    arr_nl_2.loglog(
            atoms[1:], 
            #la_pw[1:], 
            np.exp(intc_la_pw)*np.array(atoms[1:])**sp_la_pw,
            color= 'tab:blue', 
            ls = '-.',
            label = 'PW Basis (${\\alpha}$ = %5.3f)' % sp_la_pw
            )

    arr_nl_2.loglog(
            atoms[1:], 
            la[1:], 
            color= 'tab:orange', 
            marker = '*',
            label = 'Gaussian (${\\alpha}$ = %5.3f)' % sp_la
            )

    arr_nl_2.loglog(
            atoms[1:], 
            la_dg1[1:], color = 'tab:green', 
            marker = '^',
            label = 'DG $10^{-1}$ (${\\alpha}$ = %5.3f)' % sp_la_dg1
            )

    arr_nl_2.loglog(
            atoms[1:], 
            la_dg2[1:], 
            color = 'tab:red', 
            marker = '^',
            label = 'DG $10^{-2}$ (${\\alpha}$ = %5.3f)'% sp_la_dg2
            )

    arr_nl_2.loglog(
            atoms[1:13], 
            la_dg3[1:], 
            color = 'tab:purple', 
            marker = '^',
            label = 'DG $10^{-3}$ (${\\alpha}$ = %5.3f)' % sp_la_dg3
            )

    arr_nl_2.set_xlabel('Number of hydrogens')
    arr_nl_2.set_ylabel('$\lambda$')
    #arr_nl_2.set_ylim(1e2,3e5)
    arr_nl_2.set_xlim(3,32)
    arr_nl_2.xaxis.set_ticks([4,8,16,32])
    arr_nl_2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    arr_nl_2.spines['top'].set_visible(False)
    arr_nl_2.spines['right'].set_visible(False)
    #arr_nl_2.legend(frameon=False, 
    #                loc ='center left',
    #                bbox_to_anchor=(0, 0.34, 0.5, 0.5))

    arr_nl_2.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=2,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )
    fig_nl_2.tight_layout()
    fig_nl_2.subplots_adjust(top=0.79, bottom = .1)
    
    
    arr.plot(atoms, [j/i for i,j in zip(atoms, nao_dg1)],
            color = 'tab:green', marker = '^', label = 'DG $10^{-1}$')

    arr.plot(atoms[:], [j/i for i,j in zip(atoms[:], nao_dg2)],
            color = 'tab:red', marker = '^', label = 'DG $10^{-2}$')

    arr.plot(atoms[:13], [j/i for i,j in zip(atoms[:13], nao_dg3)],
            color = 'tab:purple', marker = '^', label = 'DG $10^{-3}$')
    arr.set_xlabel('Number of hydrogens')
    arr.set_ylabel('Mean VdG basis functions per atom')
    #arr.set_ylim(9,23)
    arr.set_xlim(0,32)
    arr.xaxis.set_ticks([5,10,15,20,25,30])
    arr.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    arr.spines['top'].set_visible(False)
    arr.spines['right'].set_visible(False)
    arr.legend(frameon=False, loc = 4) 



    plt.show()









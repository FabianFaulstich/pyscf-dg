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
    atoms_long = np.linspace(2,100, num = 100//2) 
    # Loewdin
    #SVD tol: 0.1
    nao = np.array([ 
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
        ])
    nao_dg1 = np.array([ 
        20, 52, 82, 112, 145, 176, 208, 240, 
        271, 303, 333, 363, 395, 426, 456 
        ])
    nnz_eri = np.array([ 
        3152.0, 50280.0, 248536.0, 705936.0, 1462844.0, 2517132.0, 
        3828361.0, 5429792.0, 7210616.0, 9162284.0, 11257588.0, 
        13482736.0, 15823456.0, 18259852.0, 20780432.0
        ])
    nnz_eri_dg1 = np.array([ 
        17712.0, 187644.0, 559780.0, 1066750.0, 1920897.0, 2719112.0,
        3419112.0, 4558044.0, 5807737.0, 6938177.0, 8015409.0, 
        9108137.0, 10668933.0, 12145132.0, 13490120.0
        ])
    la = np.array([ 
        71.75477029677036, 289.41631519330207, 626.3428606319118, 
        1079.8088281053151, 1657.6263082785426, 2395.6588234135716, 
        3495.6588234135716, 4295.604574083028, 5506.114334176364, 
        6912.050073153971, 8611.09072481073, 10473.874028508248, 
        12589.456922732159, 14975.898327909226, 17648.30604940352
        ])
    la_dg1 = np.array([ 
        435.70663782044346, 2178.9069205046203, 4671.076815098226, 
        7695.355143729995, 12166.383374696996, 17684.75248702751, 
        25284.75248702751, 32718.33665596406,  42698.31167490133, 
        54493.47205332327, 68469.46922841891, 83779.2214850416, 
        102683.77067597637, 123613.83264290619, 146644.62576291492
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
        17712.0, 443428.0, 1645420.0, 3342503.0, 5850984.0, 8448201.0, 
        11235337.0, 14513780.0, 18581269.0, 22767749.0, 26849428.0,
        31180169.0, 36723913.0, 41776357.0, 46852820.0
        ])
    la_dg2 = np.array([
        435.70663782044346, 3821.157630867125, 9665.20532786065, 
        16135.617392071692, 25428.440976715567, 36494.56007811452, 
        49797.93111841457, 66088.29724896712, 86186.85172732678,
        110122.91577847264, 138871.96522981394, 169314.0302241479, 
        207842.77643404048, 248253.30329604138, 293188.7831802176
        ])

    #SVD tol: 0.001
    nao_dg3 = np.array([
        20, 72, 124, 176, 226, 276, 326, 376, 
        426, 476, 526, 576, 626
        ])
    nnz_eri_dg3 = np.array([
        17712.0, 688860.0, 3178024.0, 6862258.0, 11331084.0, 16164052.0, 
        21140960.0, 26716800.0, 33185648.0, 40251532.0, 46625544.0, 
        53337648.0, 61619140.0
        ])
    la_dg3 = np.array([ 
        435.70663782044346, 5356.809313150105, 14721.353664060534, 
        25132.69974358416, 38279.447360792816, 54928.93498398044, 
        74344.18076190555, 97337.91128085833, 125528.32851440173, 
        158735.4410606874, 197688.2290987219, 240669.90660367682,
        291411.77167094656
        ])




    fig,      arr      = plt.subplots(nrows=1, ncols=1,  figsize=(6,4))
    fig_nl_1, arr_nl_1 = plt.subplots(nrows=1, ncols=1,  figsize=(6,5))
    fig_nl_2, arr_nl_2 = plt.subplots(nrows=1, ncols=1,  figsize=(6,5))
    fig_nl_3, arr_nl_3 = plt.subplots(nrows=1, ncols=1,  figsize=(5,5))

    #fig.tight_layout()  
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
    
    #sp_la_pw = 1.46
    #intc_la_pw = 19
    sp_la, intc_la, _, _, _= stats.linregress(
            np.log(np.array(atoms[4:])),
            np.log(np.array(la[4:]))
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


    arr_nl_3.loglog(atoms_long,
                   np.exp(intc_pw)*np.array(atoms_long)**sp_pw, 'k--')

    arr_nl_3.loglog(atoms_long,
                   np.exp(intc_mo)*np.array(atoms_long)**sp_mo, 'k--')

    arr_nl_3.loglog(atoms_long,
                   np.exp(intc)*np.array(atoms_long)**sp, 'k--')

    arr_nl_3.loglog(atoms_long,
                   np.exp(intc_dg1)*np.array(atoms_long)**sp_dg1,
                   'k--')

    arr_nl_3.loglog(atoms_long,
                   np.exp(intc_dg2)*np.array(atoms_long)**sp_dg2,
                   'k--')

    arr_nl_3.loglog(atoms_long,
                   np.exp(intc_dg3)*np.array(atoms_long)**sp_dg3,
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
            la_pw[1:], 
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
    arr.set_ylabel('DG-V basis functions per atom')
    #arr.set_ylim(9,23)
    arr.set_xlim(0,32)
    arr.xaxis.set_ticks([5,10,15,20,25,30])
    arr.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    arr.spines['top'].set_visible(False)
    arr.spines['right'].set_visible(False)
    arr.legend(frameon=False, 
               loc ='bottom right')

    plt.show()









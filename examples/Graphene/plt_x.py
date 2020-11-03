import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({'font.size': 16})

emf_small = -5.41093795805206
emf_mid   = -5.456072742748795
emf_large = -5.461170055614247

emp2_small = -0.042548330450259686
emp2_mid   = -0.04787700699095099
emp2_large = -0.049587175452327895

tol = [0.4, 0.2, 0.1, 0.05]
nb_small = [10.0,  14.5, 15.75, 16.0]
nb_mid   = [11.375, 19.75, 26.375, 30.0]
nb_large = [12.125, 20.125, 28.3125, 36.875]

emf_dg_small = [-0.17506636032566725,
                -4.577557652363815,
                -5.325093496461521,
                -5.414838363372288]

emp2_dg_small = [-0.025295137028520286,
                 -0.040606256681612656,
                 -0.043972162507642215,
                 -0.046969533504067296]

emf_dg_mid = [3.4942498518014773,
              -3.4534357057776113,
              -5.076324643278569,
              -5.418524083041677]


emp2_dg_mid = [-0.027808513765276544,
               -0.043381459885170226,
               -0.0625928161140527,
               -0.06320538571104155]

emf_dg_large = [-0.2637756893437704,
                -4.02517237269693,
                -5.052443576190973,
                -5.399705182859767]

emp2_dg_large = [-0.03666449168652824,
                 -0.05698814425288628,
                 -0.07442409774943744,
                 -0.08072122667352387]

fig0, ar0 = plt.subplots(nrows=1, ncols=1,  figsize=(7,4))
fig1, ar1 = plt.subplots(nrows=1, ncols=1,  figsize=(7,4))
fig2, ar2 = plt.subplots(nrows=1, ncols=1,  figsize=(7,4))

#tol = np.flip(np.array(tol))

ar0.plot(tol, nb_small, '-*', label='4 atoms')
ar0.plot(tol, nb_mid, '-o', label='8 atoms')
ar0.plot(tol, nb_large, '-s', label='17 atoms')
ar0.set_xlabel('DG-V trunc.')
ar0.legend(frameon=False)
ar0.spines['top'].set_visible(False)
ar0.spines['right'].set_visible(False)
ar0.set_xscale('log')
ar0.invert_xaxis()
ar0.set_xticks([4e-1, 2e-1, 1e-1, 6e-2, 5e-2])
ar0.xaxis.set_minor_locator(mtick.NullLocator())
ar0.get_xaxis().get_major_formatter().labelOnlyBase = False
ar0.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ar0.ticklabel_format(axis = 'x',
                     style = 'sci',
                     scilimits = (4,4))

ar0.set_ylabel('DG-V basis functions per atom')



ar1.plot(tol, [emf_small]*len(tol), color = 'tab:blue', marker = '*', ls = '--', 
        label='4 atoms (gth-szv)')
ar1.plot(tol, [emf_mid]*len(tol), color = 'tab:orange', marker = 'o', ls = '--', 
        label='8 atoms (gth-szv)')
ar1.plot(tol, [emf_large]*len(tol), color = 'tab:green', marker = 's', ls = '--',
        label='16 atoms (gth-szv)')

ar1.plot(tol, emf_dg_small, color = 'tab:blue', marker = '*', label='4 atoms (DG-V)')
ar1.plot(tol, emf_dg_mid, color = 'tab:orange', marker = 'o', ls = '-', label='8 atoms (DG-V)')
ar1.plot(tol, emf_dg_large, color = 'tab:green', marker = 's', ls = '-', label='16 atoms (DG-V)')
ar1.set_xlabel('DG-V trunc.')
ar1.set_ylabel('Energy (a.u.)')
ar1.spines['top'].set_visible(False)
ar1.spines['right'].set_visible(False)
ar1.invert_xaxis()
ar1.set_xscale('log')
ar1.set_xticks([4e-1, 2e-1, 1e-1, 6e-2, 5e-2])
ar1.xaxis.set_minor_locator(mtick.NullLocator())
ar1.get_xaxis().get_major_formatter().labelOnlyBase = False
ar1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ar1.ticklabel_format(axis = 'x',
                     style = 'sci',
                     scilimits = (4,4))




ar2.plot(tol, [emp2_small]*len(tol), color = 'tab:blue', marker = '*', ls = '--', 
        label='4 atoms (gth-szv)')
ar2.plot(tol, [emp2_mid]*len(tol), color = 'tab:orange', marker = 'o', ls = '--',
        label='8 atoms (gth-szv)')
ar2.plot(tol, [emp2_large]*len(tol), color = 'tab:green', marker = 's', ls = '--',
        label='16 atoms (gth-szv)')
ar2.plot(tol, emp2_dg_small, color = 'tab:blue', marker = '*', ls = '-', 
        label='4 atoms (DG-V)')
ar2.plot(tol, emp2_dg_mid, color = 'tab:orange', marker = 'o', ls = '-',
        label='8 atoms (DG-V)')
ar2.plot(tol, emp2_dg_large, color = 'tab:green', marker = 's', ls = '-',
        label='16 atoms (DG-V)')
ar2.set_xlabel('DG-V trunc.')
ar2.set_ylabel('Energy (a.u.)')
ar2.spines['top'].set_visible(False)
ar2.spines['right'].set_visible(False)
ar2.invert_xaxis()
#ar2.set_xticks([4e-1, 2e-1, 1e-1, 6e-2])
#ar2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ar2.set_xscale('log')
ar2.set_xticks([4e-1, 2e-1, 1e-1, 6e-2, 5e-2])
ar2.xaxis.set_minor_locator(mtick.NullLocator())
ar2.get_xaxis().get_major_formatter().labelOnlyBase = False
ar2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ar2.ticklabel_format(axis = 'x',
                     style = 'sci',
                     scilimits = (4,4))


ar1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=2,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )
ar2.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                                   ncol=2,
                                   frameon=False,
                                   mode = 'expand',
                                   borderaxespad=0.
                                   )

fig0.tight_layout()
fig1.tight_layout()
fig1.subplots_adjust(top=0.73, bottom = .14)
fig2.tight_layout()
fig2.subplots_adjust(top=0.73, bottom = .14)
plt.show()


'''
AO Basis              : gth-szv
Mean-field energy     : -5.465542484699777
MP2 corr. energy      : -0.05110164598057649
---------------------------------------------
SVD tollerance        : 0.4
Number of VdG bfs     : 10.541666666666666
Mean-field energy (DG): -0.37544700913261647
MP2 corr. energy  (DG): -0.0550498738249422
---------------------------------------------
SVD tollerance        : 0.2
Number of VdG bfs     : 19.166666666666668
Mean-field energy (DG): -4.202749607933096
MP2 corr. energy  (DG): -0.061958242291180056
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':

    angles = np.array([0., 0.26179939, 0.52359878, 0.78539816, 1.04719755, 1.30899694, 1.57079633])
    fig , axarr  = plt.subplots(nrows=3, ncols=3,  figsize=(20,20))
    fig1, axarr1 = plt.subplots(nrows=2, ncols=3,  figsize=(20,20))
    fig2, axarr2 = plt.subplots(nrows=2, ncols=3,  figsize=(20,20))
    fig3, axarr3 = plt.subplots(nrows=2, ncols=3,  figsize=(20,20))

    # Basis: aug-ccpvdz
    # Built in
    
    mfe_augcc    = np.array([-1.94771094, -2.07977544, -2.13079569, -2.15282300, -2.16196054, -2.16534436, -2.16621248])
    max_ev_augcc = np.array([ 9.95270540,  9.79938162,  9.66085377,  9.55712128,  9.49283581,  9.46072025,  9.45136052])
    min_ev_augcc = np.array([ 2.05985100e-05, 3.90934683e-05, 5.04521515e-05, 4.56917772e-05, 2.53210000e-05,
                              7.42538074e-06, 1.55434035e-08])
    con_no_augcc = np.array([ 4.83175987e+05, 2.50665445e+05, 1.91485467e+05, 2.09165015e+05, 3.74899719e+05, 
                              1.27410574e+06, 6.08062481e+08])
    nnz_ev_augcc = np.array([0., 0., 0., 0., 0., 1., 1.])
    
    # DG
    
    mfe_dg_augcc    = np.array([-1.94831504, -2.08014913, -2.13120463, -2.15321898, -2.16242668, -2.16596233, -2.16615613])
    max_ev_dg_augcc = np.array([1., 1., 1., 1., 1., 1., 1.])    
    min_ev_dg_augcc = np.array([1., 1., 1., 1., 1., 1., 1.])
    con_no_dg_augcc = np.array([1., 1., 1., 1., 1., 1., 1.])
    nnz_ev_dg_augcc = np.array([0., 0., 0., 0., 0., 0., 0.])

    # Basis: 631++g
    # Built in
 
    mfe_631g    = np.array([-1.94380168, -2.06237884, -2.11804747, -2.14252494, -2.15307076, -2.15723215, -2.15862764])
    max_ev_631g = np.array([ 8.02656984,  7.84921314,  7.72033673,  7.64327917,  7.60728882,  7.59567108,  7.59374386])
    min_ev_631g = np.array([ 3.29093445e-04, 4.02084601e-04, 3.78298726e-04, 3.02743020e-04, 1.94758009e-04,
                             6.82286237e-05, 3.99832950e-07])
    con_no_631g = np.array([ 24389.94140548, 19521.29756011, 20408.04316895, 25246.75608344, 39060.21041801,
                             111326.75221664, 18992291.31523133])
    nnz_ev_631g = np.array([ 0., 0., 0., 0., 0., 0., 1.])

    # DG
  
    mfe_dg_631g    = np.array([-1.93600547, -2.07302506, -2.12519329, -2.14793309, -2.15745463, -2.16170665, -2.15804249])
    max_ev_dg_631g = np.array([1., 1., 1., 1., 1., 1., 1.])    
    min_ev_dg_631g = np.array([1., 1., 1., 1., 1., 1., 1.])
    con_no_dg_631g = np.array([1., 1., 1., 1., 1., 1., 1.])
    nnz_ev_dg_631g = np.array([0., 0., 0., 0., 0., 0., 0.])

    # Basis: 321++g
    # Built in
 
    mfe_321g    = np.array([-1.94500568, -2.06354754, -2.11894539, -2.1432211, -2.15372932, -2.15789993, -2.15902687])
    max_ev_321g = np.array([ 7.74681909,  7.58158676,  7.46378213,  7.3941926,  7.36202761,  7.35183793,  7.35021526])
    min_ev_321g = np.array([ 3.83418251e-04, 4.68698990e-04, 4.37742404e-04, 3.44013351e-04, 2.13639408e-04,
                             7.10507631e-05, 4.43528118e-07])
    con_no_321g = np.array([ 2.02046175e+04, 1.61758120e+04, 1.70506262e+04, 2.14939116e+04, 3.44600637e+04,
                             1.03473033e+05, 1.65721517e+07])
    nnz_ev_321g = np.array([0., 0., 0., 0., 0., 0., 1.])

    # DG
  
    mfe_dg_321g    = np.array([-1.93416564, -2.07224805, -2.12456838, -2.14729845, -2.15681977, -2.16082043, -2.15902866])
    max_ev_dg_321g = np.array([1., 1., 1., 1., 1., 1., 1.])
    min_ev_dg_321g = np.array([1., 1., 1., 1., 1., 1., 1.])
    con_no_dg_321g = np.array([1., 1., 1., 1., 1., 1., 1.])
    nnz_ev_dg_321g = np.array([0., 0., 0., 0., 0., 0., 0.])

    # Basis: ccpvdz 
    # Built in

    mfe_cc    = np.array([-1.96647471, -2.07886256, -2.13009247, -2.15222442, -2.16140966, -2.16482107, -2.16569417])
    max_ev_cc = np.array([ 4.75966062, 4.40246204, 4.1609331, 4.01412518, 3.93221087, 3.89272481, 3.88126849])
    min_ev_cc = np.array([ 0.03053339,  0.03623688, 0.03376365, 0.0268211, 0.01821801, 0.01118896, 0.00848684])
    con_no_cc = np.array([ 155.88381208, 121.49120074, 123.23704735, 149.66297171, 215.84189799, 347.90763057, 457.32813801])
    nnz_ev_cc = np.array([0., 0., 0., 0., 0., 0., 0.])

    # DG

    mfe_dg_cc    = np.array([-1.94593146, -2.0794083,  -2.13065795, -2.15275682, -2.16198976, -2.16548826, -2.16564008])
    max_ev_dg_cc = np.array([1., 1., 1., 1., 1., 1., 1.])
    min_ev_dg_cc = np.array([1., 1., 1., 1., 1., 1., 1.])
    con_no_dg_cc = np.array([1., 1., 1., 1., 1., 1., 1.])
    nnz_ev_dg_321g = np.array([0., 0., 0., 0., 0., 0., 0.])

    
    # Basis: 631G
    # Built in

    mfe_631    = np.array([-1.90880016, -2.06097472, -2.11721567, -2.14182065, -2.15231217, -2.15639315, -2.1576372])
    max_ev_631 = np.array([ 4.05703337, 3.75567871, 3.57049889, 3.46808152, 3.41568526, 3.3921946, 3.38571429])
    min_ev_631 = np.array([ 0.05437406, 0.06427985, 0.06142294, 0.05187882, 0.04020352, 0.03092089, 0.02739527])
    con_no_631 = np.array([ 74.61339395,  58.42699498,  58.12972785,  66.84966374,  84.95985593,
                            109.70560046, 123.58753001])
    nnz_ev_631 = np.array([ 0., 0., 0., 0., 0., 0., 0.])

    # DG

    mfe_dg_631    = np.array([-1.92885635, -2.07049827, -2.12352032, -2.1465734, -2.15623523, -2.160434, -2.15706018])
    max_ev_dg_631 = np.array([1., 1., 1., 1., 1., 1., 1.])
    min_ev_dg_631 = np.array([1., 1., 1., 1., 1., 1., 1.])
    con_no_dg_631 = np.array([1., 1., 1., 1., 1., 1., 1.])
    nnz_ev_dg_631 = np.array([0., 0., 0., 0., 0., 0., 0.])

    
    # Basis: 321G
    # Built in

    mfe_321    = np.array([-1.93766069, -2.05966953, -2.11615565, -2.14071974, -2.15116652, -2.15518628, -2.15618729])
    max_ev_321 = np.array([ 3.83293843, 3.53668768, 3.36374952, 3.27150021, 3.22483135, 3.20369342, 3.19777547])
    min_ev_321 = np.array([ 0.06599903, 0.07737624, 0.07428244, 0.06395, 0.0518001, 0.04241906, 0.03890277])
    con_no_321 = np.array([ 58.07568108, 45.70767, 45.28323953, 51.15715762, 62.25531488, 75.52484865, 82.19916428])
    nnz_ev_321 = np.array([ 0., 0., 0., 0., 0., 0., 0.])
    
    # DG

    mfe_dg_321    = np.array([-1.92101408, -2.06640888, -2.1202855, -2.14368517, -2.15354427, -2.15757657, -2.15618899])
    max_ev_dg_321 = np.array([1., 1., 1., 1., 1., 1., 1.])
    min_ev_dg_321 = np.array([1., 1., 1., 1., 1., 1., 1.])
    con_no_dg_321 = np.array([1., 1., 1., 1., 1., 1., 1.])
    nnz_ev_dg_321 = np.array([0., 0., 0., 0., 0., 0., 0.])

    
    axarr[0,0].plot(180* angles/ np.pi, mfe_augcc, 'b-v', label = 'aug-ccpvdz')
    axarr[0,0].plot(180* angles/ np.pi, mfe_dg_augcc, 'r-v', label = 'DG-aug-ccpvdz')
    axarr[0,0].set_title('Hartree Fock in aug-ccpvdz')
    axarr[0,0].legend()

    axarr[2,0].plot(180* angles/ np.pi, mfe_augcc-mfe_dg_augcc, 'r-x', label = 'aug-ccpvdz')
    axarr[2,0].set_title('Difference: Built in - VDG ')
    axarr[2,0].legend()
    
    axarr1[0,0].semilogy(180* angles/ np.pi, max_ev_augcc, 'b-v', label = 'aug-ccpvdz')
    axarr1[0,0].semilogy(180* angles/ np.pi, max_ev_dg_augcc, 'r-v', label = 'DG-aug-ccpvdz')
    axarr1[0,0].set_title('Maximial Eigenvalue in aug-ccpvdz')
    axarr1[0,0].legend()

    axarr2[0,0].semilogy(180* angles/ np.pi, min_ev_augcc, 'b-v', label = 'aug-ccpvdz')
    axarr2[0,0].semilogy(180* angles/ np.pi, min_ev_dg_augcc, 'r-v', label = 'DG-aug-ccpvdz')
    axarr2[0,0].set_title('Minimial Eigenvalue in aug-ccpvdz')
    axarr2[0,0].legend()

    axarr3[0,0].semilogy(180* angles/ np.pi, con_no_augcc, 'b-v', label = 'aug-ccpvdz')
    axarr3[0,0].semilogy(180* angles/ np.pi, con_no_dg_augcc, 'r-v', label = 'DG-aug-ccpvdz')
    axarr3[0,0].set_title('Condition Number in aug-ccpvdz')
    axarr3[0,0].legend()

    axarr[1,0].plot(180* angles/ np.pi, mfe_cc, 'b-v', label = 'ccpvdz')
    axarr[1,0].plot(180* angles/ np.pi, mfe_dg_cc, 'r-v', label = 'DG-ccpvdz')
    axarr[1,0].set_title('Hartree Fock in ccpvdz')
    axarr[1,0].legend()

    axarr[2,0].plot(180* angles/ np.pi, mfe_cc-mfe_dg_cc, 'b-x', label = 'ccpvdz')
    axarr[2,0].set_title('Difference: Built in - VDG ')
    axarr[2,0].legend()

    axarr1[1,0].semilogy(180* angles/ np.pi, max_ev_cc, 'b-v', label = 'ccpvdz')
    axarr1[1,0].semilogy(180* angles/ np.pi, max_ev_dg_cc, 'r-v', label = 'DG-ccpvdz')
    axarr1[1,0].set_title('Maximial Eigenvalue in ccpvdz')
    axarr1[1,0].legend()

    axarr2[1,0].semilogy(180* angles/ np.pi, min_ev_cc, 'b-v', label = 'ccpvdz')
    axarr2[1,0].semilogy(180* angles/ np.pi, min_ev_dg_cc, 'r-v', label = 'DG-ccpvdz')
    axarr2[1,0].set_title('Minimial Eigenvalue in ccpvdz')
    axarr2[1,0].legend()

    axarr3[1,0].semilogy(180* angles/ np.pi, con_no_cc, 'b-v', label = 'ccpvdz')
    axarr3[1,0].semilogy(180* angles/ np.pi, con_no_dg_cc, 'r-v', label = 'DG-ccpvdz')
    axarr3[1,0].set_title('Condition Number in ccpvdz')
    axarr3[1,0].legend()


    axarr[0,1].plot(180* angles/ np.pi, mfe_631g, 'b-v', label = '6-31++G')
    axarr[0,1].plot(180* angles/ np.pi, mfe_dg_631g, 'r-v', label = 'DG-6-31++G')
    axarr[0,1].set_title('Hartree Fock in 631++G')
    axarr[0,1].legend() 

    axarr[2,1].plot(180* angles/ np.pi, mfe_631g-mfe_dg_631g, 'r-x', label = '6-31++G')
    axarr[2,1].set_title('Difference: Built in - VDG ')
    axarr[2,1].legend()
    
    axarr1[0,1].semilogy(180* angles/ np.pi, max_ev_631g, 'b-v', label = '6-31++G')
    axarr1[0,1].semilogy(180* angles/ np.pi, max_ev_dg_631g, 'r-v', label = 'DG-6-31++G')
    axarr1[0,1].set_title('Maximial Eigenvalue in 6-31++G')
    axarr1[0,1].legend()

    axarr2[0,1].semilogy(180* angles/ np.pi, min_ev_631g, 'b-v', label = '6-31++G')
    axarr2[0,1].semilogy(180* angles/ np.pi, min_ev_dg_631g, 'r-v', label = 'DG-6-31++G')
    axarr2[0,1].set_title('Minimial Eigenvalue in 6-31++G')
    axarr2[0,1].legend()

    axarr3[0,1].semilogy(180* angles/ np.pi, con_no_631g, 'b-v', label = '6-31++G')
    axarr3[0,1].semilogy(180* angles/ np.pi, con_no_dg_631g, 'r-v', label = 'DG-6-31++G')
    axarr3[0,1].set_title('Condition Number in 6-31++G')
    axarr3[0,1].legend()


    axarr[1,1].plot(180* angles/ np.pi, mfe_631, 'b-v', label = '6-31G')
    axarr[1,1].plot(180* angles/ np.pi, mfe_dg_631, 'r-v', label = 'DG-6-31G')
    axarr[1,1].set_title('Hartree Fock in 6-31G')
    axarr[1,1].legend()

    axarr[2,1].plot(180* angles/ np.pi, mfe_631-mfe_dg_631, 'b-x', label = '6-31G')
    axarr[2,1].set_title('Difference: Built in - VDG ')
    axarr[2,1].legend()

    axarr1[1,1].semilogy(180* angles/ np.pi, max_ev_631, 'b-v', label = '6-31G')
    axarr1[1,1].semilogy(180* angles/ np.pi, max_ev_dg_631, 'r-v', label = 'DG-6-31')
    axarr1[1,1].set_title('Maximial Eigenvalue in 6-31G')
    axarr1[1,1].legend()

    axarr2[1,1].semilogy(180* angles/ np.pi, min_ev_631, 'b-v', label = '6-31G')
    axarr2[1,1].semilogy(180* angles/ np.pi, min_ev_dg_631, 'r-v', label = 'DG-6-31G')
    axarr2[1,1].set_title('Minimial Eigenvalue in 6-31G')
    axarr2[1,1].legend()

    axarr3[1,1].semilogy(180* angles/ np.pi, con_no_631, 'b-v', label = '6-31G')
    axarr3[1,1].semilogy(180* angles/ np.pi, con_no_dg_631, 'r-v', label = 'DG-6-31G')
    axarr3[1,1].set_title('Condition Number in 6-31G')
    axarr3[1,1].legend()


    axarr[0,2].plot(180* angles/ np.pi, mfe_321g, 'b-v', label = '3-21++G')
    axarr[0,2].plot(180* angles/ np.pi, mfe_dg_321g, 'r-v', label = 'DG-3-21++G')
    axarr[0,2].set_title('Hartree Fock in 3-21++G')
    axarr[0,2].legend()

    axarr[2,2].plot(180* angles/ np.pi, mfe_321g-mfe_dg_321g, 'r-x', label = '3-21++G')
    axarr[2,2].set_title('Difference: Built in - VDG ')
    axarr[2,2].legend()

    axarr1[0,2].semilogy(180* angles/ np.pi, max_ev_321g, 'b-v', label = '3-21++G')
    axarr1[0,2].semilogy(180* angles/ np.pi, max_ev_dg_321g, 'r-v', label = 'DG-3-21++G')
    axarr1[0,2].set_title('Maximial Eigenvalue in 3-21++G')
    axarr1[0,2].legend()

    axarr2[0,2].semilogy(180* angles/ np.pi, min_ev_321g, 'b-v', label = '3-21++G')
    axarr2[0,2].semilogy(180* angles/ np.pi, min_ev_dg_321g, 'r-v', label = 'DG-3-21++G')
    axarr2[0,2].set_title('Minimial Eigenvalue in 3-21++G')
    axarr2[0,2].legend()

    axarr3[0,2].semilogy(180* angles/ np.pi, con_no_321g, 'b-v', label = '3-21++G')
    axarr3[0,2].semilogy(180* angles/ np.pi, con_no_dg_321g, 'r-v', label = 'DG-3-21++G')
    axarr3[0,2].set_title('Condition Number in 3-21++G')
    axarr3[0,1].legend()
    

    axarr[1,2].plot(180* angles/ np.pi, mfe_321, 'b-v', label = '3-21G')
    axarr[1,2].plot(180* angles/ np.pi, mfe_dg_321, 'r-v', label = 'DG-3-21G')
    axarr[1,2].set_title('Hartree Fock in 3-21G')
    axarr[1,2].legend()

    axarr[2,2].plot(180* angles/ np.pi, mfe_321-mfe_dg_321, 'b-x', label = '3-21G')
    axarr[2,2].set_title('Difference: Built in - VDG ')
    axarr[2,2].legend()

    axarr1[1,2].semilogy(180* angles/ np.pi, max_ev_321, 'b-v', label = '3-21G')
    axarr1[1,2].semilogy(180* angles/ np.pi, max_ev_dg_321, 'r-v', label = 'DG-3-21G')
    axarr1[1,2].set_title('Maximial Eigenvalue in 3-21G')
    axarr1[1,2].legend()

    axarr2[1,2].semilogy(180* angles/ np.pi, min_ev_321, 'b-v', label = '3-21G')
    axarr2[1,2].semilogy(180* angles/ np.pi, min_ev_dg_321, 'r-v', label = 'DG-3-21G')
    axarr2[1,2].set_title('Minimial Eigenvalue in 3-21G')
    axarr2[1,2].legend()

    axarr3[1,2].semilogy(180* angles/ np.pi, con_no_321, 'b-v', label = '3-21G')
    axarr3[1,2].semilogy(180* angles/ np.pi, con_no_dg_321, 'r-v', label = 'DG-3-21G')
    axarr3[1,2].set_title('Condition Number in 3-21G')
    axarr3[1,1].legend()



    plt.show()


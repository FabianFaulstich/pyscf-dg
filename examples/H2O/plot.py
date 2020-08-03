import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':

    angles = np.linspace(0, np.pi, num=20)
    angles = angles[0:16]

    mfe_dg = np.array([-22.14920674, -22.12603713, -22.54428989, -22.70160612, -22.72312158,
        -22.70157837, -22.74045598, -22.75884136, -22.76018198, -22.75865865,
        -22.74147344, -22.74882893, -22.71607919, -22.62926438, -22.43312236,
        -22.0836729 ])
    mfe_vdg = np.array([-22.55745038, -22.72258379, -22.79084372, -22.81283372, -22.78227194,
        -22.75326292, -22.74197499, -22.74340283, -22.75050797, -22.73535622,
        -22.74621853, -22.76358278, -22.79766626, -22.80827425, -22.76014639,
        -22.66485495])


    #plt.plot(33.129 +2* angles*180/np.pi, mfe   , 'b-v', label =  'HF  ')
    #plt.legend()
    #plt.show()

    plt.plot(33.129 +2* angles*180/np.pi, mfe_dg, 'r-v', label =  'HF  dzp-DG)')
    plt.plot(33.129 +2* angles*180/np.pi, mfe_vdg, 'g--v', label =  'HF  dzp-VDG)')
    plt.legend()
    plt.show()

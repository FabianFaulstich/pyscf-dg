import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func1(x, a, b, c):
    return a * np.exp(-b * x) + c

def func2(x,Y,A):
    return Y + A * x**(-3)

def func3(x, Y, C, g):
    return Y + C*(x + 1)*np.exp(-g*np.sqrt(x))

if __name__ == '__main__':

    angles = np.linspace(0, np.pi, num=20)
    angles = angles[0:16]
   

    mfe_dzp = np.array([-16.25769841, -16.44625716, -16.5249874,  -16.56098556, -16.56663014,
        -16.54998327, -16.52157846, -16.49626319, -16.49000414, -16.50737631,
        -16.53633753, -16.56047395, -16.56713158, -16.54741299, -16.49265415,
        -16.3780338])

    mfe_tzp = np.array([-16.27089976, -16.44931409, -16.52434544, -16.55975673, -16.56575353,
        -16.55002824, -16.52281916, -16.49847785, -16.49245396, -16.50916942,
        -16.536978, -16.56000099, -16.56600501, -16.5462785,  -16.49325574,
        -16.38497106])

    mfe_qzp = np.array([-16.2923636, -16.46856236, -16.54313269, -16.57829674, -16.58417902,
        -16.56829775, -16.54083312, -16.51624537, -16.51015841, -16.52703618,
        -16.55512734, -16.57835565, -16.58448261, -16.56491869, -16.5122357,
        -16.40474139])

    #Fitting CBS:
    CBS = np.zeros((2,len(mfe_dzp)))
    x = np.array([2,3,4])
    for i in range(len(mfe_dzp)):
        s = np.array([mfe_dzp[i], mfe_tzp[i], mfe_qzp[i]])
        popt, pcov = curve_fit(func3, x, s, maxfev = 1000)
        CBS[0][i] = (popt[0])
        print(popt)

        popt, pcov = curve_fit(func2, x, s)
        CBS[1][i] = (popt[0])

    mfe = np.array([-16.258583,-16.43679579, -16.51166732, -16.54685394, -16.55252775,
        -16.53640969, -16.50880487, -16.48410086, -16.47793762, -16.49499289,
        -16.5231131,  -16.54656634, -16.55284592, -16.53339476, -16.48064319,
        -16.37248727])

    mfe_dg = np.array([-16.7073792, -16.89188935, -16.96670978, -17.00222446, -17.02649595,
        -17.01948955, -17.0209126,  -17.00829744, -17.00525409, -17.01132619,
        -17.02500963, -17.0281736,  -17.01815664, -16.99329527, -16.92713021,
        -16.82963386])

    mfe_vdg = np.array([-16.7471375, -16.91606386, -16.98541296, -17.01256543, -17.0271327,
        -17.0136137,  -16.97912046, -16.98181531, -16.98421476, -16.98094704,
        -17.00766696, -17.02424028, -17.02442192, -17.00837421, -16.95672896,
        -16.8573354])

    print("Mean field in tzp:")
    print("  Minimum:", np.amin(mfe))
    print("  Maximum:", np.amax(mfe[5:-6]))
    print("  abs diff:", np.abs(np.amin(mfe) - np.amax(mfe[5:-6])))
    print("Mean field in CBS3:")
    print("  Minimum:", np.amin(CBS[0]))
    print("  Maximum:", np.amax(CBS[0][5:-6]))
    print("  abs diff:", np.abs(np.amin(CBS[0]) - np.amax(CBS[0][5:-6])))
    print("Mean field in CBS1:")
    print("  Minimum:", np.amin(CBS[1]))
    print("  Maximum:", np.amax(CBS[1][5:-6]))
    print("  abs diff:", np.abs(np.amin(CBS[1]) - np.amax(CBS[1][5:-6])))

    print("Mean field in tzp-DG:")
    print("  Maximum:", np.amin(mfe_dg))
    print("  Minimum:", np.amax(mfe_dg[5:-6]))
    print("  Abs diff:", np.abs(np.amin(mfe_dg) - np.amax(mfe_dg[5:-6])))
    print("Mean field in qzp-VDG:")
    print("  Maximum:", np.amin(mfe_vdg))
    print("  Minimum", np.amax(mfe_vdg[5:-6]))
    print("  Abs diff:", np.abs(np.amin(mfe_vdg) - np.amax(mfe_vdg[5:-6])))


    plt.plot(33.129 +2* angles*180/np.pi, mfe   , 'b-v', label =  'HF (tzp)')
    plt.plot(33.129 +2* angles*180/np.pi, CBS[0]   , 'c--v', label =  'HF (CBS3)')
    plt.plot(33.129 +2* angles*180/np.pi, CBS[1]   , 'm--v', label =  'HF (CBS1)')
    #plt.legend()
    #plt.show()

    #plt.plot(33.129 +2* angles*180/np.pi, CBS[1]   , 'm--v', label =  'HF (CBS1)')
    #plt.plot(33.129 +2* angles*180/np.pi, CBS[0]   , 'c--v', label =  'HF (CBS2)')
    plt.plot(33.129 +2* angles*180/np.pi, mfe_dg, 'r-v', label =  'HF  (tzp-DG)')
    plt.plot(33.129 +2* angles*180/np.pi, mfe_vdg, 'g--v', label =  'HF  (tzp-VDG)')
    plt.legend()
    plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(x,Y,a):
    return Y + a * x**(-3)

def get_cmp(n, name = 'hsv'):
    return plt.cm.get_cmap(name, n)


if __name__ == '__main__':


    bs  = False
    bd  = False
    mf  = False
    bi  = False
    f   = open("BS_energies.txt", "r")
    end = False

    bl  = []
    ct  = 0

    fig, axarr   = plt.subplots(nrows=3, ncols=3,  figsize=(20,20))
    fig1, axarr1 = plt.subplots(nrows=3, ncols=3,  figsize=(20,20))
    fig2, axarr2 = plt.subplots(nrows=3, ncols=3,  figsize=(20,20))
    cmap = get_cmp(4)

    for cnt, line in enumerate(f):
        if bs:
            BS = line
            bs = False
        if bd:
            BD = line
            bd = False

        if "Box size" in line:
            bs = True
        if "Box discretization" in line:
            bd = True
        if "Bondlength" in line:
            fl = [x for x in line.split()]
            h  = fl[1]
            h = h[1:]
            h = h[:-1]
            bl.append(h)

        if " DZP " in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_dzp = [float(x) for x in h]

        if "DZP-DG" in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_dg_dzp = [float(x) for x in h]

        if "DZP-VDG" in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_vdg_dzp = [float(x) for x in h] 

        if " TZP " in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_tzp = [float(x) for x in h]

        if "TZP-DG" in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_dg_tzp = [float(x) for x in h]
        
        if "TZP-VDG" in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_vdg_tzp = [float(x) for x in h]
        
        if " QZP " in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_qzp = [float(x) for x in h]
        
        if "QZP-DG" in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_dg_qzp = [float(x) for x in h]
        
        if "QZP-VDG" in line:
            end = True
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_vdg_qzp = [float(x) for x in h]
        
        if end:

            mf_cbs = np.zeros(len(mf_dzp))
            X_cbs = np.array([2,3,4])
            for i in range(len(mf_cbs)):
                s = np.array([mf_dzp[i],mf_tzp[i],mf_qzp[i]])
                popt, pcov = curve_fit(func, X_cbs, s)
                mf_cbs[i] = popt[0]

            axarr[ct, 0].plot(bl, mf_dzp    , 'b-o', label = "DZP")
            axarr[ct, 0].plot(bl, mf_dg_dzp , 'r-^', label = "DG-DZP")
            axarr[ct, 0].plot(bl, mf_vdg_dzp, 'm-v', label = "VDG-DZP")
            axarr[ct, 0].plot(bl, mf_cbs    , 'k-x', label = "CBS")
            title = BS # + " [" + BD + " ]"
            axarr[ct, 0].set_title(title)
            axarr[ct, 0].legend()

            axarr[ct, 1].plot(bl, mf_tzp    , 'b-o', label = "TZP")
            axarr[ct, 1].plot(bl, mf_dg_tzp , 'r-^', label = "DG-TZP")
            axarr[ct, 1].plot(bl, mf_vdg_tzp, 'm-v', label = "VDG-TZP")
            axarr[ct, 1].plot(bl, mf_cbs    , 'k-x', label = "CBS")
            title = BS # + " [" + BD + " ]"
            axarr[ct, 1].set_title(title)
            axarr[ct, 1].legend()

            axarr[ct, 2].plot(bl, mf_qzp    , 'b-o', label = "QZP")
            axarr[ct, 2].plot(bl, mf_dg_qzp , 'r-^', label = "DG-QZP")
            axarr[ct, 2].plot(bl, mf_vdg_qzp, 'm-v', label = "VDG-QZP")
            axarr[ct, 2].plot(bl, mf_cbs    , 'k-x', label = "CBS")
            title = BS # + " [" + BD + " ]"
            axarr[ct, 2].set_title(title)
            axarr[ct, 2].legend()

            axarr2[ct, 0].plot(bl, mf_dzp-mf_cbs    , 'b-o', label = "diff-DZP")
            axarr2[ct, 0].plot(bl, mf_dg_dzp-mf_cbs , 'r-^', label = "diff-DG-DZP")
            axarr2[ct, 0].plot(bl, mf_vdg_dzp-mf_cbs, 'm-v', label = "diff-VDG-DZP")
            title = BS # + " [" + BD + " ]"
            axarr2[ct, 0].set_title(title)
            axarr2[ct, 0].legend()

            axarr2[ct, 1].plot(bl, mf_tzp-mf_cbs    , 'b-o', label = "diff-TZP")
            axarr2[ct, 1].plot(bl, mf_dg_tzp-mf_cbs , 'r-^', label = "diff-DG-TZP")
            axarr2[ct, 1].plot(bl, mf_vdg_tzp-mf_cbs, 'm-v', label = "diff-VDG-TZP")
            title = BS # + " [" + BD + " ]"
            axarr2[ct, 1].set_title(title)
            axarr2[ct, 1].legend()

            axarr2[ct, 2].plot(bl, mf_qzp-mf_cbs    , 'b-o', label = "diff-QZP")
            axarr2[ct, 2].plot(bl, mf_dg_qzp-mf_cbs , 'r-^', label = "diff-DG-QZP")
            axarr2[ct, 2].plot(bl, mf_vdg_qzp-mf_cbs, 'm-v', label = "diff-VDG-QZP")
            title = BS # + " [" + BD + " ]"
            axarr2[ct, 2].set_title(title)
            axarr2[ct, 2].legend()
            
            
            axarr1[0, 0].plot(bl, mf_dzp, color = cmap(ct), marker = 'o', label = BS)
            axarr1[0, 0].set_title("DZP") 
            axarr1[0, 0].legend()

            axarr1[1, 0].plot(bl, mf_tzp, color = cmap(ct), marker='o', label = BS)
            axarr1[1, 0].set_title("TZP")
            axarr1[1, 0].legend()

            axarr1[2, 0].plot(bl, mf_qzp, color = cmap(ct), marker='o', label = BS)
            axarr1[2, 0].set_title("QZP")
            axarr1[2, 0].legend()

            axarr1[0, 1].plot(bl, mf_vdg_dzp, color = cmap(ct), marker='v', label = BS)
            axarr1[0, 1].set_title("VDG-DZP")
            axarr1[0, 1].legend()

            axarr1[1, 1].plot(bl, mf_vdg_tzp, color = cmap(ct), marker='v', label = BS)
            axarr1[1, 1].set_title("VDG-TZP")
            axarr1[1, 1].legend()

            axarr1[2, 1].plot(bl, mf_vdg_qzp, color = cmap(ct), marker='v', label = BS)
            axarr1[2, 1].set_title("VDG-QZP")
            axarr1[2, 1].legend()

            axarr1[0, 2].plot(bl, mf_dg_dzp, color = cmap(ct), marker='^', label = BS)
            axarr1[0, 2].set_title("DG-DZP")
            axarr1[0, 2].legend()

            axarr1[1, 2].plot(bl, mf_dg_tzp, color = cmap(ct), marker='^', label = BS)
            axarr1[1, 2].set_title("DG-TZP")
            axarr1[1, 2].legend()

            axarr1[2, 2].plot(bl, mf_dg_qzp, color = cmap(ct), marker='^', label = BS)
            axarr1[2, 2].set_title("DG-QZP")
            axarr1[2, 2].legend()

            #print(mf_dzp)
            #print(mf_dg_dzp)
            #print(mf_vdg_dzp)
            #print(mf_tzp)
            #print(mf_dg_tzp)
            #print(mf_vdg_tzp)
            #print(mf_qzp)
            #print(mf_dg_qzp)
            #print(mf_vdg_qzp)
            #print(mf_cbs)
            
            ct += 1 
            bl  = []
            end = False     
    plt.show()


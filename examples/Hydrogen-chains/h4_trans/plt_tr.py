import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(x,Y,a):
    return Y + a * x**(-3)

def get_cmp(n, name = 'hsv'):
    return plt.cm.get_cmap(name, n)


if __name__ == '__main__':


    bs = False
    mf = False
    mf_iter = False
    mf_dg_iter = False
    rt = False
    angle = False
    angle_iter = False
    mf_plt = True
    con = False
    catch_cond = True

    f   = open("out_tr.txt", "r")
    cmap = get_cmp(9)

    fig, arccpvdz = plt.subplots(nrows=2, ncols=2,  figsize=(20,20))
    fig1, ar6311  = plt.subplots(nrows=2, ncols=3,  figsize=(20,20))
    fig2, ar631   = plt.subplots(nrows=2, ncols=3,  figsize=(20,20))
    fig3, ar321   = plt.subplots(nrows=2, ncols=2,  figsize=(20,20))

    fig4, arcon   = plt.subplots(nrows=2, ncols=2,  figsize=(20,20))


    for cnt, line in enumerate(f):

        if bs:
            BS = line
            bs = False
        
        if angle_iter:
            angle_iter = False
            fl = [x for x in line.split()]
            fl[-1] = fl[-1][:-1]
            rot_a += [float(fl[0])]


        if angle:
            fl = [x for x in line.split()]
            #fl[0] = fl[0][1:]
            fl = fl[1:] # neglecting first point
            if fl[-1] == "]":
                fl = fl[:-1]
            else:
                fl[-1] = fl[-1][:-1]
            rot_a = [float(x) for x in fl]
            angle = False
            angle_iter = True
        
        
        if rt:
            relt = line
            rt = False

        if 'Rotation angles:' in line:
            angle = True
        
        if 'AO basis' in line:
            bs = True

        if 'Relative truncation' in line:
            rt = True

        

        if mf_iter:
            mf_iter = False
            fl = [x for x in line.split()]
            fl[-1] = fl[-1][:-1]
            mf += [float(fl[0])] 
            if mf_plt:
                #plot
                if 'ccpvdz' in BS:
                    if len(BS) > 7:
                        arccpvdz[0,0].plot(rot_a, mf[1:], 'D-k' ,label = BS)
                        arccpvdz[0,0].set_title("aug-cc-pVDZ")
                        arccpvdz[0,0].legend()
                    elif len(BS) < 8:
                        arccpvdz[0,1].plot(rot_a, mf[1:], 'D-k' ,label = BS)
                        arccpvdz[0,1].set_title("cc-pVDZ")
                        arccpvdz[0,1].legend()
                elif '6311' in BS:
                    if len(BS) == 8:
                        ar6311[0,0].plot(rot_a, mf[1:], 'D-k' ,label = BS)
                        ar6311[0,0].set_title("6-311++G")
                        ar6311[0,0].legend()
                    elif len(BS) == 7:
                        ar6311[0,1].plot(rot_a, mf[1:], 'D-k' ,label = BS) 
                        ar6311[0,1].set_title("6-311+G")
                        ar6311[0,1].legend()
                    elif len(BS) == 6:
                        ar6311[0,2].plot(rot_a, mf[1:], 'D-k' ,label = BS)
                        ar6311[0,2].set_title("6-311G")
                        ar6311[0,2].legend()
                elif '631' in BS and '11' not in BS:
                    if len(BS) == 7:
                        ar631[0,0].plot(rot_a, mf[1:], 'D-k' ,label = BS)
                        ar631[0,0].set_title("6-31++G")
                        ar631[0,0].legend()
                    elif len(BS) == 6:
                        ar631[0,1].plot(rot_a, mf[1:], 'D-k' ,label = BS)
                        ar631[0,1].set_title("6-31+G")
                        ar631[0,1].legend()
                    elif len(BS) == 5:
                        ar631[0,2].plot(rot_a, mf[1:], 'D-k' ,label = BS)
                        ar631[0,2].set_title("6-31G")
                        ar631[0,2].legend()

                elif '321' in BS:
                    if len(BS) == 7:
                        ar321[0,0].plot(rot_a, mf[1:], 'D-k' ,label = BS)
                        ar321[0,0].set_title("3-21++G")
                        ar321[0,0].legend()
                    elif len(BS) == 5:
                        ar321[0,1].plot(rot_a, mf[1:], 'D-k' ,label = BS)
                        ar321[0,1].set_title("3-21G")
                        ar321[0,1].legend()

                mf_plt = False
       
        if 'Mean-field energy:' in line:
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf = [float(x) for x in h]
            mf_iter = True

        if con and catch_cond:
            fl = [x for x in line.split()]
            fl[-1] = fl[-1][:-1]
            if fl[-1] == '':
                fl = fl[:-1]
            cond +=  [float(x) for x in fl]
            con = False
            catch_cond = False
            #plot
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    arcon[0,0].semilogy(rot_a, cond[1:], 'D-k' ,label = BS)
                    arcon[0,0].set_title("aug-cc-pVDZ")
                    arcon[0,0].legend()
                elif len(BS) < 8:
                    arcon[0,0].semilogy(rot_a, cond[1:], 'r-x' ,label = BS) 
                    arcon[0,0].set_title("cc-pVDZ")
                    arcon[0,0].legend()
            elif '6311' in BS: 
                if len(BS) == 8:
                    arcon[0,1].semilogy(rot_a, cond[1:], 'D-k' ,label = BS) 
                    arcon[0,1].set_title("6-311++G")
                    arcon[0,1].legend()
                elif len(BS) == 7:
                    arcon[0,1].semilogy(rot_a, cond[1:], 'x-r' ,label = BS) 
                    arcon[0,1].set_title("6-311+G")
                    arcon[0,1].legend()
                elif len(BS) == 6:
                    arcon[0,1].semilogy(rot_a, cond[1:], '+-g' ,label = BS) 
                    arcon[0,1].set_title("6-311G")
                    arcon[0,1].legend()
            elif '631' in BS and '11' not in BS: 
                if len(BS) == 7:
                    arcon[1,0].semilogy(rot_a, cond[1:], 'D-k' ,label = BS) 
                    arcon[1,0].set_title("6-31++G")
                    arcon[1,0].legend()
                elif len(BS) == 6:
                    arcon[1,0].semilogy(rot_a, cond[1:], 'r-x' ,label = BS) 
                    arcon[1,0].set_title("6-31+G")
                    arcon[1,0].legend()
                elif len(BS) == 5:
                    arcon[1,0].semilogy(rot_a, cond[1:], 'g-+' ,label = BS) 
                    arcon[1,0].set_title("6-31G")
                    arcon[1,0].legend()
            elif '321' in BS: 
                if len(BS) == 7:
                    arcon[1,1].semilogy(rot_a, cond[1:], 'D-k' ,label = BS) 
                    arcon[1,1].set_title("3-21++G")
                    arcon[1,1].legend()
                elif len(BS) == 5:
                    arcon[1,1].semilogy(rot_a, cond[1:], 'r-x' ,label = BS) 
                    arcon[1,1].set_title("3-21G")
                    arcon[1,1].legend()





        if 'Condition no.:' in line and catch_cond:
            cond_line = line 
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            elif h[0] == '':
                h = h[1:]
            #else:
                #h[-1] = h[-1][:-1]
            cond = [float(x) for x in h]
            con = True


        if mf_dg_iter:
            mf_dg_iter = False
            fl = [x for x in line.split()]
            fl[-1] = fl[-1][:-1]
            mf_dg += [float(fl[0])]
            if 'ccpvdz' in BS:
                if len(BS) > 7:
                    arccpvdz[0,0].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    arccpvdz[1,0].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    arccpvdz[0,0].legend()
                elif len(BS) < 8:
                    arccpvdz[0,1].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    arccpvdz[1,1].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    arccpvdz[0,1].legend()

            elif '6311' in BS: 
                if len(BS) == 8:
                    ar6311[0,0].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x' ,label =  "DG-" + relt) 
                    ar6311[1,0].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    ar6311[0,0].set_title("6-311++G")
                    ar6311[0,0].legend()
                elif len(BS) == 7:
                    ar6311[0,1].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x' ,label =   "DG-" + relt)
                    ar6311[1,1].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    ar6311[0,1].set_title("6-311+G")
                    ar6311[0,1].legend()
                elif len(BS) == 6:
                    print(BS)
                    ar6311[0,2].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x' ,label =   "DG-" + relt)
                    ar6311[1,2].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    ar6311[0,2].set_title("6-311G")
                    ar6311[0,2].legend()
            
            elif '631' in BS and '11' not in BS:
                if len(BS) == 7:
                    ar631[0,0].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x' ,label =  "DG-" + relt)
                    ar631[1,0].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    ar631[0,0].set_title("6-31++G")
                    ar631[0,0].legend()
                elif len(BS) == 6:
                    ar631[0,1].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x' ,label =   "DG-" + relt)
                    ar631[1,1].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    ar631[0,1].set_title("6-31+G")
                    ar631[0,1].legend()
                elif len(BS) == 5:
                    ar631[0,2].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x' ,label =   "DG-" + relt)
                    ar631[1,2].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    ar631[0,2].set_title("6-31G")
                    ar631[0,2].legend()

            elif '321' in BS:
                if len(BS) == 7:
                    ar321[0,0].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x' ,label =   "DG-" + relt)
                    ar321[1,0].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    ar321[0,0].set_title("3-21++G")
                    ar321[0,0].legend()
                elif len(BS) == 5:
                    ar321[0,1].plot(rot_a, mf_dg[1:], color = cmap(dgc), marker= 'x' ,label =   "DG-" + relt)
                    ar321[1,1].semilogy(rot_a, [a-b for a,b in zip(mf[1:],mf_dg[1:])], color = cmap(dgc), marker= 'x',label = "DG-" + relt)
                    ar321[0,1].set_title("3-21G")
                    ar321[0,1].legend()

        if 'Mean-field energy (DG):' in line:
            fl = [x for x in line.split()]
            h  = fl[3:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            else:
                h[-1] = h[-1][:-1]
            mf_dg = [float(x) for x in h]
            dgc += 1
            mf_dg_iter = True

        if 'AO basis:' in line:
            mf_plt = True
            catch_cond = True
            dgc = 0
    #plt.show()

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker 

import numpy as np
from scipy.optimize import curve_fit

def func(x,Y,a):
    return Y + a * x**(-3)

def get_cmp(n, name = 'hsv'):
    return plt.cm.get_cmap(name, n)


if __name__ == '__main__':


    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"    
    plt.rcParams.update({'font.size': 16})

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

    fig0, arcon0 = plt.subplots(nrows=1, ncols=1,  figsize=(5,3))
    fig1, arcon1 = plt.subplots(nrows=1, ncols=1,  figsize=(5,3))
    fig2, arcon2 = plt.subplots(nrows=1, ncols=1,  figsize=(5,3))
    fig3, arcon3 = plt.subplots(nrows=1, ncols=1,  figsize=(5,3))


    for cnt, line in enumerate(f):

        if bs:
            BS = line
            bs = False
        
        if angle_iter:
            angle_iter = False
            fl = [x for x in line.split()]
            fl[-1] = fl[-1][:-1]
            rot_a += [180.0/np.pi*float(fl[0])]


        if angle:
            fl = [x for x in line.split()]
            #fl[0] = fl[0][1:]
            fl = fl[1:] # neglecting first point
            if fl[-1] == "]":
                fl = fl[:-1]
            else:
                fl[-1] = fl[-1][:-1]
            rot_a = [180.0*float(x)/np.pi for x in fl]
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
                    arcon0.semilogy(rot_a, cond[1:], 'o-r' ,
                            label = "aug-cc-pVDZ")
                
                elif len(BS) < 8:
                    arcon0.semilogy(rot_a, cond[1:], 'g-+' ,
                            label = "cc-pVDZ")
                    arcon0.set_ylim(1e1,1e10)
                    arcon0.xaxis.set_ticks([15,30,45,60,75,90])
                    arcon0.set_ylabel('Condition number')
                    arcon0.set_xlabel('Angle ($\\alpha$) in degree')
                    arcon0.spines['top'].set_visible(False)
                    arcon0.spines['right'].set_visible(False)
                    arcon0.legend(loc = 2, frameon=False)
                    fig0.tight_layout()
            elif '6311' in BS: 
                if len(BS) == 8:
                    arcon1.semilogy(rot_a, cond[1:], 'o-r' ,
                            label = "6-311++G") 
                
                elif len(BS) == 6:
                    arcon1.semilogy(rot_a, cond[1:], '+-g' ,
                            label = "6-311G") 
                    arcon1.set_ylim(1e1,1e10)
                    arcon1.xaxis.set_ticks([15,30,45,60,75,90])
                    arcon1.set_ylabel('Condition number')
                    arcon1.set_xlabel('Angle ($\\alpha$) in degree')
                    arcon1.spines['top'].set_visible(False)
                    arcon1.spines['right'].set_visible(False)
                    arcon1.legend(loc = 2, frameon=False) 
                    fig1.tight_layout()
            elif '631' in BS and '11' not in BS: 
                if len(BS) == 7:
                    arcon2.semilogy(rot_a, cond[1:], 'o-r' ,
                            label = "6-31++G") 
                
                elif len(BS) == 5:
                    arcon2.semilogy(rot_a, cond[1:], 'g-+' ,
                            label = "6-31G") 
                    arcon2.set_ylim(1e1,1e10)
                    arcon2.xaxis.set_ticks([15,30,45,60,75,90])
                    arcon2.set_ylabel('Condition number')
                    arcon2.set_xlabel('Angle ($\\alpha$) in degree')
                    arcon2.spines['top'].set_visible(False)
                    arcon2.spines['right'].set_visible(False)
                    arcon2.legend(loc = 2, frameon=False)
                    fig2.tight_layout()
            elif '321' in BS: 
                if len(BS) == 7:
                    arcon3.semilogy(rot_a, cond[1:], 'o-r' ,
                            label = "3-21++G") 
                elif len(BS) == 5:
                    arcon3.semilogy(rot_a, cond[1:], 'g-+' ,
                            label = "3-21G") 
                    arcon3.set_ylim(1e1,1e10)
                    arcon3.xaxis.set_ticks([15,30,45,60,75,90])
                    arcon3.set_ylabel('Condition number')
                    arcon3.set_xlabel('Angle ($\\alpha$) in degree')
                    arcon3.spines['top'].set_visible(False)
                    arcon3.spines['right'].set_visible(False)
                    arcon3.legend(loc = 2, frameon=False)
                    fig3.tight_layout()




        if 'Condition no.:' in line and catch_cond:
            cond_line = line 
            fl = [x for x in line.split()]
            h  = fl[2:]
            h[0] = h[0][1:]
            if h[-1] == "]":
                h = h[:-1]
            elif h[0] == '':
                h = h[1:]
            cond = [float(x) for x in h]
            con = True


        if mf_dg_iter and '0.5' not in relt:
            mf_dg_iter = False
            fl = [x for x in line.split()]
            fl[-1] = fl[-1][:-1]
            mf_dg += [float(fl[0])]
            diff = np.array([a-b for a,b in zip(mf[1:],mf_dg[1:])])
            mini=np.amin(diff)
            mini=1000*np.round(mini,4)
            maxi=np.amax(diff)
            maxi=1000*np.round(maxi,4)
            
        if 'Mean-field energy (DG):' in line and '0.5' not in relt:
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
    
    plt.show()

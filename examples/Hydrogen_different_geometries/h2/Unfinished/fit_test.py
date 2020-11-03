import numpy as np
import scipy
from scipy.optimize import curve_fit

def func(x,Y,a):
    return Y + a * x**(-3)

if __name__ == '__main__':
    X_cbs = np.array([2,3,4])
    s = np.array([-1.02133, -1.04015, -1.04259])
    popt, pcov = curve_fit(func, X_cbs, s)
    print(popt)

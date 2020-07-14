import numpy as np


def unfold(m,n):
    l = m % n
    k = (m-l)/n
    return int(l), int(k)

def unfold_sym(n):
    k = 1 
    while n>(1+k)*k/2.0:
        k = k+1
    l = n - (k-1)*k/2.0
    return int(k-1), int(l-1)

def get_red_idx(n):
    idx_sym = np.zeros(int(n*(n+1)/2));
    count   = 0;
    for j in range(n):
        l = n-j;
        idx_sym[count:count+l] = np.arange(n*j+j, n*j+n);
        count = count + l;
    return idx_sym.astype(int)




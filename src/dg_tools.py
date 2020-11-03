import math
import numpy as np
from scipy.spatial import Delaunay, Voronoi
import matplotlib.pyplot as plt
from numpy import linalg as la
import copy

from pyscf.pbc import tools

# For NNZ-ERI and Lambda values

def naive_voronoi(grid, atoms, dx, dy, dz):

    per = np.array([-1 ,0, 1])
    for i, atom in enumerate(atoms):
        dist_atom_grid = np.sum(np.abs(grid - atom)**2,axis=-1)**(1./2)
        
        # periodic distance
        for x in per:
            for y in per:
                for z in per:
                    if x == 0 and y == 0 and z == 0:
                        continue
                    else:
                        atom_p = [x*dx+atom[0], y*dy+atom[1], z*dz+atom[2]] 
                        dist_atom_grid_p = np.sum(np.abs(grid - atom_p)**2,
                                axis=-1)**(1./2)
                        dist_atom_grid = np.minimum(dist_atom_grid, 
                               dist_atom_grid_p)
        if i == 0:
            distances = dist_atom_grid
        else:
            distances = np.vstack((distances, dist_atom_grid))

    mask    = np.argsort(distances, axis=0)[0]
    idx_mat = np.zeros((len(mask), len(atoms)), dtype=bool) 
    
    mask = mask.reshape((len(mask),1))
    np.put_along_axis(idx_mat, mask, True, axis = 1)
    
    return idx_mat

def get_dg_nnz_eri_loop(cell, aoR, b_idx, exx=False):

    coords = copy.copy(cell.get_uniform_grids())
    mesh = cell.mesh
    vol = cell.vol
    ngrids = np.prod(mesh)
    assert ngrids == aoR.shape[0]
    dvol = vol / ngrids
    nao = aoR.shape[1]
    eri = np.zeros((nao,nao,nao,nao))
    vcoulR_pairs = np.zeros((ngrids,nao,nao))
    coulG = tools.get_coulG(cell, mesh=mesh, exx=exx)
    assert aoR.dtype == np.double

    print("        primitive loops ...")
    for q in range(nao):
        for s in range(nao):
            aoR_qs = aoR[:,q] * aoR[:,s]
            aoG_qs = tools.fft(aoR_qs, mesh) * dvol
            vcoulG_qs = coulG * aoG_qs
            vcoulR_pairs[:,q,s] = tools.ifft(vcoulG_qs, mesh).real / dvol
    print()
    print("        primitive loops, eneter contractions ...")
    eri = np.einsum('xp,xr,xqs->prqs', aoR, aoR, vcoulR_pairs, optimize=True) * dvol
    print()
    return np.count_nonzero(eri)


def get_dg_nnz_eri(cell, aoR, b_idx, exx=False):
    '''Generate the ERI tensor

    This is only a proof of principle and the implementation is not 
    efficient in terms of either memory or speed

    input:

    aoR: (VdG) basis projection matrix and assumed to be 
         columnwise L2-normalized 
    '''

    nnz_eri  = 0
    n_lambda = 0

    coords = copy.copy(cell.get_uniform_grids())

    print("    Treating EXX:", exx)
    mesh   = cell.mesh
    vol    = cell.vol
    ngrids = np.prod(mesh)
    assert ngrids == aoR.shape[0]
    dvol   = vol / ngrids
    nao    = aoR.shape[1]
    print("    No. of DG orbitals: ", nao)
    print("    No. of grid points: ", ngrids)

    coulG = tools.get_coulG(cell, mesh=mesh, exx=exx)

    assert aoR.dtype == np.double

    # Careful! Check memory

    print("        matricized loops with symmetry ...")

    # number of block pairs (k,l) with k <= l
    m = int(len(b_idx) * (len(b_idx) + 1)/2.0)

    for i in range(1,m+1):
        print("            Computing compressed kl-block tensor: ", i, 
                ' of ', m+1)
        
        k, l  = unfold_sym(i)
        idx_k = b_idx[k]
        idx_l = b_idx[l]

        # exploit symmetry in k:th block
        idx_k_sym = get_red_idx(len(idx_k))
        
        # Solving Poisson eq. in k:th VdG-element
        # Extracting k:th VdG basis functions 
        bl_k       = aoR[:,idx_k[0]:idx_k[-1]+1]
        
        # Repeat k:th block to speed up the basis-pair access 
        # bl_k_mat  = [1,...,N_k|1,...,N_k|...|1 ,...,N_k]
        # bl_k_perm = [1 ,..., 1|2 ,..., 2|...|N_k,...N+k]
        bl_k_mat   = np.matlib.repmat(bl_k, 1, len(idx_k))
        del bl_k
        idx_k_perm = [j * len(idx_k) + i for i in range(len(idx_k)) 
                        for j in range(len(idx_k))]
        bl_k_mat_perm = bl_k_mat[:,idx_k_perm]
                                                           
        # exploiting symmerty before using poisson solver:
        bl_k_dens     = bl_k_mat[:,idx_k_sym] * bl_k_mat_perm[:,idx_k_sym]
        del bl_k_mat, bl_k_mat_perm
        bl_k_sol_pois = np.zeros_like(bl_k_dens)

        for j in range(bl_k_dens.shape[1]):
            coulG_k = coulG * tools.fft(bl_k_dens[:,j], mesh) * dvol
            bl_k_sol_pois[:,j] = tools.ifft(coulG_k, mesh).real / dvol

        del bl_k_dens

        # exploit symmetry in l:th block
        idx_l_sym = get_red_idx(len(idx_l))

        # Computing DG-basis product in l:th DG-block
        bl_l = aoR[:,idx_l[0]:idx_l[-1]+1]
        bl_l_mat = np.matlib.repmat(bl_l, 1, len(idx_l))
        del bl_l
        idx_l_perm     = [j * len(idx_l) + i for i in range(len(idx_l))
                          for j in range(len(idx_l))]
        bl_l_mat_perm  = bl_l_mat[:,idx_l_perm]
        bl_l_mat_prod  = bl_l_mat[:,idx_l_sym] * bl_l_mat_perm[:,idx_l_sym]
        del bl_l_mat
        del bl_l_mat_perm

        # Compute integral of density times pair interaction
        eri_kl  = np.dot(bl_l_mat_prod.T,bl_k_sol_pois) * dvol
        del bl_l_mat_prod, bl_k_sol_pois

        # Creating full kl-tensor:
        # Recreating neglected and kept indices in block k and l in 
        # order to place and the computed elements in the k-l-block-tensor

        # k:th block-size is (nao_k, nao_k)
        nao_k = int(len(idx_k))

        # Creating index for elements that were neglected (neg_k).
        # At this point neg_k takes the form:
        # neg_k = [[0     , ...,    nao_k -1],
        #          [nao_k , ..., 2* nao_k -1],
        #          ...,
        #          [(nao_k-1)* nao_k, ..., nao_k**2 -1]]
        neg_k     = np.arange(nao_k**2).reshape(nao_k, nao_k)
        neg_k_per = np.copy(neg_k)
        neg_k_per = np.transpose(neg_k_per)

        # Extract lower triangular matrix of neg_k and vectorize 
        # non-zero elements:
        # neg_k = [nao_k, 
        #          2* nao_k, 2* nao_k + 1, 
        #          3* nao_k, 3* nao_k + 1, 3* nao_k + 3,  
        #          ...,
        #          (nao_k - 1)* nao_k, ...., nao_k**2 -2] 
        neg_k     = np.tril(neg_k,-1).reshape(-1)
        neg_k     = neg_k[neg_k != 0]

        # Analogously to neg_k we compute the elements that were computed except
        # the diagonal elements
        # neg_k_perm = [1,
        #               2, 102,
        #               3, 103, 203'
        #               ...,
        #               nao_k -1, nao_k + (nao_k -1), ..., (nao_k -1)* nao_k -1]
        neg_k_per = np.tril(neg_k_per,-1).reshape(-1)
        neg_k_per = neg_k_per[neg_k_per != 0]

        # The indeces that were computed above are the `negative' of neg_k,
        # the elements are the same as in neg_k_perm plus the diagonal elements.
        # Note, however, that the ordering is different.
        kep_k     = np.delete(np.arange(nao_k**2), neg_k)

        # Perform analogous computations for l:th block
        nao_l = int(len(idx_l))
        
        neg_l     = np.arange(nao_l**2).reshape((nao_l, nao_l))
        neg_l_per = np.copy(neg_l)
        neg_l_per = np.transpose(neg_l_per)

        neg_l     = np.tril(neg_l,-1).reshape(-1)
        neg_l     = neg_l[neg_l != 0]
        
        neg_l_per = np.tril(neg_l_per,-1).reshape(-1)
        neg_l_per = neg_l_per[neg_l_per != 0]
        
        kep_l     = np.delete(np.arange(nao_l**2), neg_l)

        # Generating mask for kept indices
        # kep_kl is in x major 
        kep_kl = [[x,y] for y in kep_k for x in kep_l]
        mask = np.zeros((nao_l**2, nao_k**2), dtype=bool)
        for idx in kep_kl:
            mask[idx[0],idx[1]] = True
        
        # Generate matrizised k-l-eri-block tensor 
        eri_kl_full = np.zeros((nao_l**2, nao_k**2))
        
        # place the computed eri entries based at the positions defined mask
        np.place(eri_kl_full, mask, eri_kl)

        # Copying elements that were not computed within k:th and l:th block,
        # respectively. Note that neg_k_per/ neg_l_per does not include 
        # diagonal elements, hence, no double counting.
        eri_kl_full[:,neg_k] = eri_kl_full[:,neg_k_per]
        eri_kl_full[neg_l,:] = eri_kl_full[neg_l_per,:]

        # Neglect ERI that are absolute smaller than 1e-6 
        eri_kl_full = np.abs(eri_kl_full)
        eri_kl_full[eri_kl_full <= 1e-6] = 0

        if k != l:
            nnz_eri  += 2*np.count_nonzero(eri_kl_full)
            n_lambda += 2*np.sum(np.abs(eri_kl_full))
        else:
            nnz_eri  += np.count_nonzero(eri_kl_full)
            n_lambda += np.sum(np.abs(eri_kl_full))
    return n_lambda, nnz_eri     

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

def get_red_idx_nd(n):
    idxarr = []
    for i in range(n-1):
        idx = i * n + np.arange(i+1,n)
        idxarr += idx.tolist()
    return idxarr

def get_red_idx(n):
    idx_sym = np.zeros(int(n*(n+1)/2));
    count   = 0;
    for j in range(n):
        l = n-j;
        idx_sym[count:count+l] = np.arange(n*j+j, n*j+n);
        count = count + l;
    return idx_sym.astype(int)


def get_cmap(n, name='hsv'):
    '''coloring for visualize'''
    return plt.cm.get_cmap(name, n)


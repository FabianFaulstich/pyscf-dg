import numpy as np
import numpy.matlib
from numpy import linalg as la
from scipy.linalg import block_diag
import scipy

import pyscf
from pyscf import lib
from pyscf.pbc import dft as dft
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import FFTDF
from pyscf.pbc import tools
from pyscf import ao2mo
from sys import exit 
import time
from pyscf.fci import direct_spin0, direct_spin1

import dg_tools

class dg_model_ham:
    def __init__(self, cell):
        
        self.cell = cell

        print("    Fetching uniform grid ...")
        start = time.time()
        self.coords   = cell.get_uniform_grids()
        end = time.time()
        print("    Done! Elapsed time: ", end - start, "sec.")
        print()

        print("    Computing DG-Gramm matrix ...")
        start = time.time()
        self.dg_gramm, self.dg_idx = get_dg_gramm(self.cell)
        end = time.time()
        print("    Done! Elapsed time: ", end - start, "sec.")
        print()

        self.nao      = self.dg_gramm.shape[1]
        
        print("    Computing overlap matrix ...")
        start = time.time()
        self.ovl_dg   = np.einsum('xi,xj->ij', self.dg_gramm,
                                  self.dg_gramm) * cell.vol / np.prod(cell.mesh)
        end = time.time()
        print("    Done! Elapsed time: ", end - start, "sec.")
        print()
        
        self.cell_dg      = gto.Cell()
        self.cell_dg.atom = cell.atom
        self.cell_dg.a    = cell.a
        self.cell_dg.ovl  = self.ovl_dg
        self.cell_dg.unit = 'B'
        
        self.cell_dg      = self.cell_dg.build()
        self.cell_dg.ovl  = self.ovl_dg

        self.cell_dg.pbc_intor = lambda *arg, **kwargs: self.ovl_dg
        
        self.cell_dg.pbc_intor = lambda *arg, **kwargs: self.ovl_dg
                
        print("    Computing kinetic-energy matrix ...")
        start = time.time()
        self.kin_dg   = get_kin_numint_G(cell, self.coords, self.dg_gramm)
        end = time.time()
        print("    Done! Elapsed time: ", end - start, "sec.")
        print()
        
        print("    Computing nucleon interaction ...")
        start = time.time()
        self.nuc_dg   = get_pp_numint(cell, self.coords, self.dg_gramm)
        end = time.time()
        print("    Done! Elapsed time: ", end - start, "sec.")
        print()

        self.hcore_dg = self.kin_dg + self.nuc_dg
        
        # ERI.
        #
        # If exx=None, then the results would matches that from PySCF exactly.
        #   Note that this implies that PySCF does not handle the G=0 term
        #   at the level of the ERIs.
        #
        # Presumably the correct treatment should be exx='ewald', but this
        # means that the contribution from the Madelung constant should
        # appear elsewhere in the total energy etc.
        print("    Computing ERI ...")
        start = time.time()
        self.eri = get_eri(cell, self.coords, self.dg_gramm, self.dg_idx, exx=None)
        end = time.time()
        print("    Done! Elapsed time: ", end - start, "sec.")
        print()

    def run_RHF(self):
        
        self.mf_dg         = scf.RHF(self.cell_dg, exxdiv='ewald')
        self.mf_dg.verbose = 3 #5
        
        dm = np.zeros((self.nao,self.nao))
        
        self.mf_dg.get_hcore = lambda *args: self.hcore_dg
        self.mf_dg.get_ovlp  = lambda *args: self.ovl_dg
        self.mf_dg._eri      = ao2mo.restore(8, self.eri, self.nao)
        
        self.mf_dg.kernel(dm0 = dm, dump_chk=False)
        self.emf = self.mf_dg.e_tot
        return self.emf

    def run_MP2(self):

        self.mf_dg.with_df._numint.eval_ao = lambda *args:(self.dg_gramm,0)
        self.mf_dg.with_df.mesh = self.cell.mesh # discretization points
        self.df = mp.MP2(self.mf_dg)

        self.emp_corr, _ = self.df.kernel()
        return self.emp_corr, self.emp_corr + self.emf

    def run_CC(self):

        self.cc_dg = cc.CCSD(self.mf_dg)
        self.cc_dg.kernel()
        self.ecc_corr = self.cc_dg.e_corr
        return self.ecc_corr, self.ecc_corr + self.emf


def get_kin_numint_G(cell, coords, aoR):
    '''Evaluate the kinetic energy matrix directly on the grid.

    This computes the derivatives numerically via the Fourier grid.

    The current code only supports gamma-point calculation
    '''
    mesh = cell.mesh
    vol = cell.vol
    ngrids = np.prod(mesh)
    assert ngrids == aoR.shape[0]
    dvol = vol / ngrids
    nao = aoR.shape[1]

    aoG = np.empty(aoR.shape, np.complex128)
    for i in range(nao):
        aoG[:,i] = tools.fft(aoR[:,i], cell.mesh) * dvol

    Gv = cell.get_Gv(cell.mesh)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)

    # kinetic energy in Fourier
    kin = 0.5 * np.einsum('gi,g,gj->ij', aoG.conj(), absG2, aoG) / vol

    if aoR.dtype == np.double:
        return kin.real
    else:
        return kin

def get_pp_numint(cell, coords, aoR):
    '''Evaluate the pseudopotential directly on the grid.

    This follow the code 

        pbc/df/fft.py
        pbc/gto/pseudo/test/test_pp.py

    The current code only supports gamma-point calculation
    '''
    if not cell.pseudo:
        raise RuntimeError('Must use pseudopotential')

    mesh = cell.mesh
    vol = cell.vol
    ngrids = np.prod(mesh)
    assert ngrids == aoR.shape[0]
    dvol = vol / ngrids
    nao = aoR.shape[1]
    SI = cell.get_SI()
    Gv = cell.get_Gv(mesh)
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -np.einsum('ij,ij->j', SI, vpplocG)

    # vpploc evaluated in real-space. 
    # NOTE: the convention of ifft is 1/N_g. By replacing the correct
    # factor 1/vol by 1/N_g, vpplocR implicitly carries a factor of
    #
    #   vol / N_g = dvol 
    #
    # I prefer getting rid of this, so that vpplocR is indeed the value
    # of vloc collocated on the real space grid, and multiply dvol later
    # during integration
    vpplocR = tools.ifft(vpplocG, mesh).real / dvol

    # local part evaluated in real space
    vpploc = np.einsum('xi,x,xj->ij', aoR, vpplocR, aoR) * dvol

    # nonlocal part evaluated in reciprocal space
    # this follows pbc/gto/pseudo/test/test_pp.py
    # vppnonloc evaluated in reciprocal space
    #
    # aoG means aokG with kpt being the gamma point
    #
    # NOTE: the convention of fft is 1, which misses a dvol factor
    #
    # The scaling of the quadrature is also changed, so that there is no
    # extra anomolous scaling factors in the end.
    aoG = np.empty(aoR.shape, np.complex128)
    for i in range(nao):
        aoG[:,i] = tools.fft(aoR[:,i], cell.mesh) * dvol

    vppnl = np.zeros((nao,nao), dtype=np.complex128)
    hs, projGs = pseudo.get_projG(cell)
    for ia, [h_ia,projG_ia] in enumerate(zip(hs,projGs)):
        for l, h in enumerate(h_ia):
            nl = h.shape[0]
            for m in range(-l,l+1):
                SPG_lm_aoG = np.zeros((nl,nao), dtype=np.complex128)
                for i in range(nl):
                    SPG_lmi = SI[ia,:] * projG_ia[l][m][i]
                    # NOTE: each quadrarture in the G basis should have a 1/vol factor 
                    SPG_lm_aoG[i,:] = np.einsum('g,gp->p', SPG_lmi.conj(), aoG) / vol
                for i in range(nl):
                    for j in range(nl):
                        vppnl += h[i,j]*np.einsum('p,q->pq',
                                                   SPG_lm_aoG[i,:].conj(),
                                                   SPG_lm_aoG[j,:])

    vpp = vpploc + vppnl
    if aoR.dtype == np.double:
        return vpp.real
    else:
        return vpp


def get_eri(cell, coords, aoR, b_idx, exx=None):
    '''Generate the ERI tensor

    This does not consider the 8-fold symmetry.
    
    This is only a proof of principle and the implementation is not 
    efficient in terms of either memory or speed
    '''
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

    # Careful! Check memory

    print("        matricized loops with symmetry ...")
    start = time.time()
    
    m = int(len(b_idx) * (len(b_idx) + 1)/2.0)
    eri_new_sym = np.zeros((nao,nao,nao,nao))
    for i in range(m):
        print("            Computing compressed kl-tensor ...")
        start_comp = time.time()
        k, l  = dg_tools.unfold_sym(i)
        idx_k = b_idx[k]
        idx_l = b_idx[l]
    
        # exploit symmetry in k:th block
        idx_k_sym = dg_tools.get_red_idx(len(idx_k))

        # Solving Poisson eq. in k:th DG-block
        bl_k          = aoR[:,idx_k[0]:idx_k[-1]+1]
        bl_k_mat      = np.matlib.repmat(bl_k, 1, len(idx_k))
        idx_k_perm    = [j * len(idx_k) + i for i in range(len(idx_k))
                          for j in range(len(idx_k))]
        bl_k_mat_perm = bl_k_mat[:,idx_k_perm]
        
        # exploiting symmerty before using poisson solver:
        bl_k_dens     = bl_k_mat[:,idx_k_sym] * bl_k_mat_perm[:,idx_k_sym]
        bl_k_sol_pois = np.zeros_like(bl_k_dens)
        
        for j in range(bl_k_dens.shape[1]):
            coulG_k            = coulG * tools.fft(bl_k_dens[:,j], mesh) * dvol
            bl_k_sol_pois[:,j] = tools.ifft(coulG_k, mesh).real / dvol
        
        # exploit symmetry in l:th block
        idx_l_sym = dg_tools.get_red_idx(len(idx_l))
        
        # Computing DG-basis product in l:th DG-block
        bl_l           = aoR[:,idx_l[0]:idx_l[-1]+1]
        bl_l_mat       = np.matlib.repmat(bl_l, 1, len(idx_l))
        idx_l_perm     = [j * len(idx_l) + i for i in range(len(idx_l))
                          for j in range(len(idx_l))]
        bl_l_mat_perm  = bl_l_mat[:,idx_l_perm]
        bl_l_mat_prod  = bl_l_mat[:,idx_l_sym] * bl_l_mat_perm[:,idx_l_sym]

        # Compute integral Pois. part and product part
        eri_kl  = np.dot(bl_l_mat_prod.T,bl_k_sol_pois) * dvol
        end_comp = time.time()
        print("            Done! Elapsed time: ", end_comp - start_comp, "sec.")
        print()

        # Creating full kl-tensor:
        #   Recreating neglected and kept indices 
        print("            writing compressed kl-tensor to full tensor ...")
        start_wrt = time.time()

        neg_k     = np.arange(len(idx_k)**2).reshape(int(len(idx_k)), int(len(idx_k)))
        neg_k_per = np.copy(neg_k)
        neg_k_per = np.transpose(neg_k_per)

        neg_k     = np.tril(neg_k,-1).reshape(-1)
        neg_k     = neg_k[neg_k != 0]
        neg_k_per = np.tril(neg_k_per,-1).reshape(-1)
        neg_k_per = neg_k_per[neg_k_per != 0]
        kep_k     = np.delete(np.arange(int(len(idx_k)**2)), neg_k)
        
        neg_l     = np.arange(len(idx_l)**2).reshape(int(len(idx_l)), int(len(idx_l)))
        neg_l_per = np.copy(neg_l)
        neg_l_per = np.transpose(neg_l_per)

        neg_l     = np.tril(neg_l,-1).reshape(-1)
        neg_l     = neg_l[neg_l != 0]
        neg_l_per = np.tril(neg_l_per,-1).reshape(-1)
        neg_l_per = neg_l_per[neg_l_per != 0]
        kep_l     = np.delete(np.arange(int(len(idx_l)**2)), neg_l) 

        #   Generating mask for kept indices
        kep_kl = [[x,y] for y in kep_l for x in kep_k]
        mask = np.zeros((len(idx_k)**2, len(idx_l)**2), dtype=bool)
        for idx in kep_kl:
            mask[idx[0],idx[1]] = True
        
        eri_kl_full = np.zeros((len(idx_k)**2,len(idx_l)**2))
        np.place(eri_kl_full, mask, eri_kl)
        eri_kl_full[:,neg_k] = eri_kl_full[:,neg_k_per] 
        eri_kl_full[neg_l,:] = eri_kl_full[neg_l_per,:]
        
        # store full kl-tensor 
        eri_new_sym[idx_l[0]:idx_l[-1]+1,idx_l[0]:idx_l[-1]+1,
                idx_k[0]:idx_k[-1]+1,idx_k[0]:idx_k[-1]+1] = eri_kl_full.reshape((len(idx_l),len(idx_l),len(idx_k),len(idx_k))) 
        if k != l:
            eri_new_sym[idx_k[0]:idx_k[-1]+1,idx_k[0]:idx_k[-1]+1,
                    idx_l[0]:idx_l[-1]+1,idx_l[0]:idx_l[-1]+1] = np.transpose(eri_kl_full).reshape((len(idx_l),len(idx_l),len(idx_k),len(idx_k))) 
        
        end_wrt = time.time()
        print("            Done! Elapsed time: ", end_wrt - start_wrt, "sec.")
        print()
    end = time.time()
    print("        Done! Elapsed time: ", end - start, "sec.")
    print()


    #print("        matricized loops ...")
    #start = time.time()
    
    #m = len(b_idx)**2
    #eri_new = np.zeros((nao,nao,nao,nao))
    #for i in range(m):
    #    k, l  = dg_tools.unfold(i,len(b_idx))
    #    idx_k = b_idx[k]
    #    idx_l = b_idx[l]
    #        
    #    # Solving Poisson eq. in k:th DG-block
    #    bl_k          = aoR[:,idx_k[0]:idx_k[-1]+1]
    #    bl_k_mat      = np.matlib.repmat(bl_k, 1, len(idx_k))
    #    idx_k_perm    = [j * len(idx_k) + i for i in range(len(idx_k)) 
    #                      for j in range(len(idx_k))]
    #    bl_k_mat_perm = bl_k_mat[:,idx_k_perm]
    #    bl_k_dens     = bl_k_mat * bl_k_mat_perm
    #    bl_k_sol_pois = np.zeros_like(bl_k_dens)
    #    for i in range(bl_k_dens.shape[1]):
    #        coulG_k            = coulG * tools.fft(bl_k_dens[:,i], mesh) * dvol
    #        bl_k_sol_pois[:,i] = tools.ifft(coulG_k, mesh).real / dvol    
    #
    #    # Computing DG-basis product in l:th DG-block
    #    bl_l           = aoR[:,idx_l[0]:idx_l[-1]+1]
    #    bl_l_mat       = np.matlib.repmat(bl_l, 1, len(idx_l))
    #    idx_l_perm     = [j * len(idx_l) + i for i in range(len(idx_l))
    #                      for j in range(len(idx_l))]
    #    bl_l_mat_perm  = bl_l_mat[:,idx_l_perm]
    #    bl_l_mat_prod  = bl_l_mat * bl_l_mat_perm
    #    
    #    # Compute integral Pois. part and product part
    #    eri_kl  = np.dot(bl_l_mat_prod.T,bl_k_sol_pois) * dvol
    #    eri_tsr = np.reshape(eri_kl, (len(idx_l),len(idx_l),len(idx_k),len(idx_k)))
    #    eri_new[idx_l[0]:idx_l[-1]+1,idx_l[0]:idx_l[-1]+1,
    #            idx_k[0]:idx_k[-1]+1,idx_k[0]:idx_k[-1]+1] = eri_tsr
    #    
    #end = time.time()
    #print("        Done! Elapsed time: ", end - start, "sec.")
    #print()

    #print("comparing output")
    #print("[0,0,0,0]:")
    #print(eri_new_sym[0,0,0,0])
    #print(eri_new[0,0,0,0])
    #print("[1,0,0,0]")
    #print(eri_new_sym[1,0,0,0])
    #print(eri_new[1,0,0,0])
    #print("[0,1,0,0]:")                                                         
    #print(eri_new_sym[0,1,0,0])                                                         
    #print(eri_new[0,1,0,0])                                                     
    #print("[0,0,1,0]")                                                          
    #print(eri_new_sym[0,0,1,0])                                                         
    #print(eri_new[0,0,1,0]) 
    #print("[5,2,8,4]")
    #print(eri_new_sym[5,2,8,4])                                                         
    #print(eri_new[5,2,8,4])

    #print("        primitive loops ...")
    #start = time.time()
    #for q in range(nao):
    #    for s in range(nao):
    #        aoR_qs = aoR[:,q] * aoR[:,s]
    #        aoG_qs = tools.fft(aoR_qs, mesh) * dvol
    #        vcoulG_qs = coulG * aoG_qs
    #        vcoulR_pairs[:,q,s] = tools.ifft(vcoulG_qs, mesh).real / dvol
    #end = time.time()
    #print("        Done! Elapsed time: ", end - start, "sec.")
    #print()
    #print("        primitive loops, eneter contractions ...")
    #start = time.time()
    #eri = np.einsum('xp,xr,xqs->prqs', aoR, aoR, vcoulR_pairs) * dvol
    #end = time.time()
    #print("        Done! Elapsed time: ", end - start, "sec.")
    #print()

    #print("[0,0,0,0]:")
    #print(eri[0,0,0,0])
    #print(eri_new[0,0,0,0])
    #print("[1,0,0,0]")
    #print(eri[1,0,0,0])
    #print(eri_new[1,0,0,0])
    #print("[0,1,0,0]:")                                                         
    #print(eri[0,1,0,0])                                                         
    #print(eri_new[0,1,0,0])                                                     
    #print("[0,0,1,0]")                                                          
    #print(eri[0,0,1,0])                                                         
    #print(eri_new[0,0,1,0]) 
    #print("[5,2,8,4]")
    #print(eri[5,2,8,4])                                                         
    #print(eri_new[5,2,8,4]) 
    return eri_new_sym

#def unfold(m,n):
#    l = m % n
#    k = (m-l)/n 
#    return int(l), int(k)


def get_dg_gramm(cell,svd_tol=1e-3):
    '''Generate the Gramm matrix for fake-DG basis
    '''

    coords = cell.get_uniform_grids()
    x_dp   = np.array([x[0] for x in coords])

    # Fetching Gramm matrix
    ao_values = dft.numint.eval_ao(cell, coords, deriv=0) # change to eval_gto?

    # Determine ideal DG-cuts for quasi-1D system along z-axis
    atom_pos = np.array([x[1][0] for x in cell.atom])

    iDG_cut  = atom_pos[0:-1] + np.diff(atom_pos)/2.0
    DG_cut   = np.zeros(len(iDG_cut), dtype=int)

    for i in range(len(iDG_cut)):
        DG_cut[i] = np.argmin(np.absolute(x_dp - iDG_cut[i])) # check for min being unambiguous
        
    DG_cut = np.append(0, np.append( DG_cut, len(x_dp)))
    dvol = cell.vol / np.prod(cell.mesh)
    return get_dg_basis(dvol, ao_values, DG_cut, svd_tol)


def get_dg_basis(dvol, Gr_mat,DG_cut,svd_tol):
    '''Creating (fake) DG basis
    '''

    U_out     = []
    index_out = []
    offset    = 0
    for i in range(len(DG_cut)-1):

        # Extracting DG-blocks of Gramm matrix
        dg_block = Gr_mat[DG_cut[i]:DG_cut[i+1]]
        U, S, _  = la.svd(dg_block, full_matrices=False)

        #U_Q, _   = scipy.linalg.qr(U, mode='economic', pivoting=False)
        #U = U_Q

        # Basis compression
        S    = S[::]
        S_block = S[S > svd_tol]
        U_block = U[:,0:len(S_block)]
        
        # Storing block indices
        index_out.append([offset + elem for elem in range(len(S_block))])
        offset = index_out[-1][-1]+1
        
        # Storing on block-diagonal form
        # change to U_out = [U_block(1),...]
        U_out = block_diag(U_out, U_block)
    U_out = 1/np.sqrt(dvol) * np.array(U_out[1:,:])
    return U_out, index_out


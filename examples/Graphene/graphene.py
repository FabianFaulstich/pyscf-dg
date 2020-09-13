#!/usr/bin/env python

'''
Graphene, a low dimensional PBC system

Here we compare the two methods of AFTDF and DF for this low-dimensional
PBC system.  The AFTDF integrates the non-periodic direction using gauss-
quadrature while the DF uses an analytic version of the fast-fourier transform
for this slab-like system.

Because of this, the fast-fourier transform method in the DF is generally
faster than the AFTDF, but requires many more reciprocal vectors in the
non-periodic direction.

'''

import sys
sys.path.append('../../src')
import time
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf

import dg_model_ham as dg
import dg_tools
import numpy as np

nk = 1
kpts = [nk,nk,1]
Lz = 25 # Smallest Lz value for ~1e-6 convergence in absolute energy
a = 1.42 # bond length in graphene
fft_ke_cut = 300
# Much smaller mesh needed for AFTDF with the setting cell.low_dim_ft_type='inf_vacuum'
aft_mesh = [30,30,40]
e = []
t = []
pseudo = 'gth-pade'


cell = pbcgto.Cell()
cell.build(unit = 'B',
           a = [[4.6298286730500005, 0.0, 0.0], [-2.3149143365249993, 4.009549246030899, 0.0], [0.0, 0.0, Lz]],
           atom = [['C',[0, 0, 0]], ['C', [0, 2.67303283, 0]]],
           mesh = [50,50,50],
           #dimension=2,
           #low_dim_ft_type = 'inf_vacuum',
           pseudo = pseudo,
           verbose = 3,
           #precision = 1e-6,
           basis='gth-szv')
t0 = time.time()
atoms_2d = np.array([atom[1][:2] for atom in cell.atom])
V_net = dg_tools.get_V_net_per(atoms_2d, 0, cell.a[0][0],0, cell.a[1][1])
voronoi_cells = dg_tools.get_V_cells(V_net, atoms_2d)
cell_vdg = dg.dg_model_ham(cell, None ,'rel_num', .99, True, voronoi_cells, V_net)
e.append(cell_vdg.run_RHF())
t.append(time.time() - t0)

##################################################
#
# 2D PBC with AFT
#
##################################################
cell = pbcgto.Cell()
cell.build(unit = 'B',
           a = [[4.6298286730500005, 0.0, 0.0], [-2.3149143365249993, 4.009549246030899, 0.0], [0.0, 0.0, Lz]],
           atom = 'C 0 0 0; C 0 2.67303283 0',
           mesh = aft_mesh,
           dimension=2,
           low_dim_ft_type = 'inf_vacuum',
           pseudo = pseudo,
           verbose = 3,#7
           precision = 1e-6,
           basis='gth-szv')
t0 = time.time()
mf = pbchf.KRHF(cell)
mf.with_df = pdf.AFTDF(cell)
mf.kpts = cell.make_kpts(kpts)
mf.conv_tol = 1e-6
e.append(mf.kernel())
t.append(time.time() - t0)

##################################################
#
# 2D PBC with FFT
#
##################################################
cell = pbcgto.Cell()
cell.build(unit = 'B',
           a = [[4.6298286730500005, 0.0, 0.0], [-2.3149143365249993, 4.009549246030899, 0.0], [0.0, 0.0, Lz]],
           atom = 'C 0 0 0; C 0 2.67303283 0',
           ke_cutoff = fft_ke_cut,
           dimension=2,
           pseudo = pseudo,
           verbose = 3,
           precision = 1e-6,
           basis='gth-szv')
t0 = time.time()
mf = pbchf.KRHF(cell, exxdiv='ewald')
mf.with_df = pdf.FFTDF(cell)
mf.kpts = cell.make_kpts(kpts)
mf.conv_tol = 1e-6
e.append(mf.kernel())
t.append(time.time() - t0)


##################################################
#
# 2D PBC with GDF
#
##################################################
t0 = time.time()
mf = pbchf.KRHF(cell)
mf.with_df = pdf.GDF(cell)
mf.kpts = cell.make_kpts(kpts)
mf.conv_tol = 1e-6
e.append(mf.kernel())
t.append(time.time() - t0)

##################################################
#
# 2D PBC with MDF
#
##################################################
t0 = time.time()
mf = pbchf.KRHF(cell)
mf.with_df = pdf.MDF(cell)
mf.kpts = cell.make_kpts(kpts)
mf.conv_tol = 1e-6
e.append(mf.kernel())
t.append(time.time() - t0)

print('Energy (DG) (AFTDF) (FFTDF) (GDF)   (MDF)')
print(e)
print('Timing (DG) (AFTDF) (FFTDF) (GDF)   (MDF)')
print(t)


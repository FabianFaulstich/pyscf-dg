import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py

from functools import reduce
import numpy
from pyscf import gto, scf, ao2mo, fci, mcscf
from pyscf import tools
from pyscf import symm

dist_range = numpy.arange(1.2,2.6,0.1)

energy_data = numpy.zeros((len(dist_range),3))
me_data = numpy.zeros((len(dist_range),5))
for n, dist in enumerate(dist_range):
    mol = gto.M(
    atom = [
            ['He', 0.0,  0.0,  0.0],
            ['Be', 0.0,  0.0,  dist]
           ],
    basis = 'def2-svpd',
    verbose = 0,
        symmetry = 1,
    #    symmetry_subgroup = 'D2h',
    )
    mf = scf.RHF(mol)
    mf.kernel()

    mc_mo = mf.mo_coeff
    #
    # 3. Exited states.  In this example, B2u are bright states.
    #
    # Here, mc.ci[0] is the first excited state.
    #
    mc = mcscf.CASCI(mf, 20, 6)
    mc.fcisolver = fci.direct_spin0.FCI(mol)
    mc.fcisolver.nroots = 7
    mc.kernel(mc_mo)


    energy_data[n,0] = dist
    energy_data[n,1:] = mc.e_tot[:2]
    print()
    print("Energies:", mc.e_tot[:2])
    print()
    sys.stdout.flush()
    #
    # 4. transition density matrix and transition dipole
    #
    # Be careful with the gauge origin of the dipole integrals
    #
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = numpy.einsum('z,zx->x', charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    dip_ints = mol.intor('cint1e_r_sph', comp=3)

    def makedip(ci_id1, ci_id2):
        # transform density matrix in MO representation
        t_dm1 = mc.fcisolver.trans_rdm1(mc.ci[ci_id1], mc.ci[ci_id2], mc.ncas, mc.nelecas)
        # transform density matrix to AO representation
        orbcas = mc_mo[:,mc.ncore:mc.ncore+mc.ncas]
        t_dm1_ao = reduce(numpy.dot, (orbcas, t_dm1, orbcas.T))
        # transition dipoles
        return numpy.einsum('xij,ji->x', dip_ints, t_dm1_ao)

    dip_array = numpy.zeros(4)
    count = 0
    for i in range(2):
      for j in range(2):
        dip = makedip(i,j)[0]
        print('Transition dipole between |%d> and |%d>'%(i,j), dip)
        dip_array[count] = dip
        count += 1
    sys.stdout.flush()

    me_data[n,0] = dist
    me_data[n,1:] = dip_array

numpy.savetxt('energy_data.txt', energy_data)
numpy.savetxt('me_data.txt', me_data)

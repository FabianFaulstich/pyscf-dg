# PySCF DG-Voronoi

This is a pilot version of the DG-Voronoi procedure implemented as an 
extension of pyscf.pbc.gto.Cell class. A minimal example (H2) is given by

	examples/Hydrogen_different_geometries/h2/vdg-pes.py 

showing DG-Voronoi procedure at the mean-field, MP2 and CCSD level of theory.
A DG Hamiltonian (a dg_model_ham object) is created from a given Cell 
object in the following way:

	dg_model_ham(cell, dg_cuts, dg_trunc, svd_tol, voronoi, dg_on, gram)

Input:

	cell, pyscf.pbc.gto.cell object describing the considered system
	
	dg_cuts, list describing ``cuts'' for the DG procedure with 
		 rectangular partitioning strategy. 
		 Default is set to None
	dg_trunc, string describing the truncation procedure to compress the 
		  DG basis, i.e., truncation of the singular values. 
		  Options are:
		  	rel_tol: truncating w.r.t. the relative size of the 
				 singular values
			abs_tol: truncating w.r.t. the absolute size of the 
				 singular values
			rel_num: truncating w.r.t. the relative number of the 
				 singular values
			abs_num: truncating w.r.t. the absolute number of 
				 singular values
			rel_cum: truncating w.r.t. the relative cumulative sum of 
				 the singular values 
			abs_cum: truncating w.r.t. the absolute cumulative sum of
                                 the singular values
		  Default is set to 'abs_tol'
	svd_tol, float describing the truncation tollerance for the set 
		 truncation procedure.
		 Default is set to 1e-3 for 'abs_tol' 
	voronoi, boolean describing if a Voronoi partitioning is used. If set to 
		 False, the DG-Rectangular procedure will be used. 
		 Default is set to True 
	dg_on, boolean describing if a DG procedure (DG-R or DG-V) is applied.
	       Default is set to True
	gram, numpy.array for customizing the basis projection matrix \Phi.
	      Default is set to None, i.e., the projection matrix is obtained by 
	      projecting the AO basis of the Cell object.

Given a DG Hamiltonian, RHF, MP2 and CCSD can be executed through 
	dg_model_ham.run_RHF()
	dg_model_ham.run_MP2()
	dg_model_ham.run_CC()

respectively. 

Note that this is merely a pilot implementation version of the DG-Voronoi 
(DG-Rectangular) procedure that has grown over the past months, i.e., 
input variables, dependences, algorithmic layout and code commentation are by no 
means optimal or in a final version. Older versions and testing routines are 
still included in the code (sometimes as large commented blocks). 

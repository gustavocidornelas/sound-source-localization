__all__ = ["var_sub"]

var_sub = """
.. |def_K| replace:: `K` : number of samples
.. |def_L| replace:: `L` : number of output channels if multi-output (A)LSSMs are used, ``()`` otherwise
.. |def_M_indep| replace:: `M` : Number of independent state variables
.. |def_M_input| replace:: `M` : Number of inputs of a LSSM
.. |def_M_models| replace:: `M` : Number of models
.. |def_N| replace:: `N` : (A)LSSM system order, corresponding to the number of state variables
.. |def_S| replace:: `S` : number of signal sets for parallel processing.
.. |def_P| replace:: `P` : number of segments in a composite cost (`CCost`); for single cost segments (`CostSeg`) `P=1`


.. |fct_doc_lssm_update_header| replace:: Internal update of model matrices and check of parameter validity.
.. |fct_doc_lssm_update| replace:: An explicit call of this method is commonly not required as it is called internally whenever needed. 

.. |note_MC| replace::  
   Working with **multi-channel** signals requires **multi-output** (A)LSSMs,
   which has a direct impacts on the dimensionality of many required function parameters 
   and return values used throughout this module. For more details, see also <HERE in ALSSM>. 


.. |def_poly_q| replace:: :math:`q \in \mathbb{R}^Q` : exponent vector
.. |def_poly_Q| replace:: `Q` : number of polynomial exponents/coefficients
.. |def_poly_r| replace:: :math:`r \in \mathbb{R}^R` : exponent vector
.. |def_poly_R| replace:: `R` : number of polynomial exponents/coefficients
.. |def_poly_alpha| replace:: :math:`\\alpha \in \mathbb{R}^Q` : polynomial coefficients 
.. |def_poly_beta| replace:: :math:`\\beta \in \mathbb{R}^R` : polynomial coefficients
.. |def_poly_talpha| replace:: :math:`\\tilde{\\alpha} \in \mathbb{R}^RQ` : joined polynomial coefficients 



"""

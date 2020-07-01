.. _lmlib_single_multi_input_output_lssm:

Single- and Multi-Input/Output LSSM
===================================
*lmlib* distinguishes between single and multiple inputs and outputs,
therefore LSSMs are able to handel single- and multi-chanel signals.
The definition of single- and multi-channel signals is written here: :ref:`lmlib_single_multi_channel`.

**Imporant** : The model input or/and ouput has to match with the signal.
For example, if you compare the output of an autonomous linear state space model with a single-channel signal
then the ALSSM output has to be single as well.

Here another example: If an LSSM has a multi-channel signal as input and a single-channel signal as output,
then the LSSM needs a multi-input and single-output.

How To Define Single- / Multi- Input/Output
-------------------------------------------
An ALSSM or LSSM is single-output by setting a **one-dimensional** output matrix  :math:`C`.
Whereas a multi-output ALSSM/LSSM is defined by a **two-dimensional** output matrix.

For single-input on a LSSM the input matrix :math:`B` is **one-dimensional** and
**two-dimensional** for a  multi-channel signal (multi-input).


Note
----
For multi-channel signals of shape ``(K, 1, O)``, where ``K``  is the number of samples and ``O`` the number of channels
the (A)LSSM in-/output needs to be multi-input/output.

* For example: ``C = [[1, 0, 0]]`` or ``B=[[1], [0], [0]]``
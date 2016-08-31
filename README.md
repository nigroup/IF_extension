Integrate-and-Fire neuron model extension to reproduce the dynamics of a ball-and-stick (soma + dendrite) model neuron exposed to a weak electric field
=====

This code provides an implementation of the point neuron model extension presented in the article:

**Aspart, Ladenbauer, Obermayer (2016). _Extending integrate-and-fire model neurons to account for the effects of weak electric fields and input filtering mediated by the dendrite._**

The repository contains an example code to simulate the extended LIF point (eP) neuron model in Python as well as a NEURON implementation of the ball-and-stick (BS) neuron model, both subject to fluctuating somatic and distal dendritic input currents as well as a sinusoidal extracellular field.

**How to cite this code**: If you use this code for your published research, we suggest that you cite the above mentioned article. 

Requirements
=====
Python libraries:
* Numpy
* Matplotlib
* Numba (for fast computation of the eP neuron model)
* Jupyter (or IPython Notebook)
* PyNeuron (optional, but required to simulate the BS neuron model)

Additionally, a working installation of NEURON is needed in order to run the BS model (e.g., for comparison purposes). It is, however, not necessary if one is interested in the output of the eP model only.


Content
=====
| File | Description |
|---|---|
| run_BS_and_eP_models.ipynb    | IPython notebook which generates example output time series of the BS and eP models for different (input and field) parameter values. In case NEURON is not installed, the eP model can be run independently. |
| ePModel.py                    | Python implementation of the eP neuron model. |
| BS_morphology.hoc             | Contains the NEURON implementation of the BS model (definition of morphology, extracellular field, etc.) |
| compile_mod.sh                | Script to compile the NEURON mechanisms (under Linux). The mechanisms need to be compiled in a certain order, which is performed by this script. |
| Mod_files/fsin.mod            | NEURON mechanism to generate a sinusoidal extracellular field. |
| Mod_files/IClampOU.mod        | NEURON mechanism to generate an Ornstein-Uhlenbeck process to be used as input current. | 
| Mod_files/Spikeout.mod        | NEURON spike mechanism that resets the membrane voltage after a given threshold is reached. It also included a refractory time. |
| Mod_files/xtra.mod            | NEURON mechanism to set the extracellular field, i.e., the gradient of extracellular potential. |

The mod files originate from the NEURON ModelDB and the NEURON forum and were slightly modified for the purpose of this project (see the comments in each file for more details).

**The mod files need to be compiled in a specific order**.
Please use the provided shell script (compile_mod.sh) or reuse the same compilation order as in the script if you compile under Windows.

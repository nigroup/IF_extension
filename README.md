Integrate-and-Fire neuron model extension to reproduce the dynamics of a ball-and-stick (soma + dendrite) model neuron exposed to a weak electric field
=====

This code is an implementation of the point neuron model extension presented in the article:

**Aspart, Ladenbauer, Obermayer (2016). _Extending integrate-and-fire model neurons to account for the effects of weak electric fields and input filtering mediated by the dendrite._**

This repository contains a running example of the extended LIF point neuron model (eP) in Python as well as a NEURON implementation of a ball-and-stick (BS) neuron model subject to fluctuating somatic and distal dendritic input currents as well as a sinusoidal extracellular field.

**How to cite this code**: if you use this code for your published research, we suggest that you cite the above mentioned article. 

Requirements
=====
Python libraries:
* Numpy
* Matplotlib
* Numba (for faster computation of the IF point neuron model)
* Jupyter (or IPython Notebook)
* PyNeuron (optional, but required to simulate the BS neuron model)

Additionally, you will need a working installation of NEURON in order to run the BS model. This is not necessary to run only the point neuron model extension, though.


Content
=====
| File | Description |
|---|---|
| run_BS_and_eP_models.ipynb    | IPython notebook containing example on how to run the BS and eP model. In case NEURON is not installed, the point neuron model can be run independently |
| ePModel.py                    | Python implementation of the eP neuron model |
| BS_morphology.hoc             | Contains the NEURON implementation of the BS model (morphology defition,  extracellular fields, etc...) |
| compile_mod.sh                | Script to compile the NEURON mechanism (under Linux). The mechanism needs to be compiled in a given order, therefore this script is needed |
| Mod_files/fsin.mod            | NEURON mechanism: for a bogus point process containing a variables which oscillates in time. It is used to create a sinusoidal extracellular field|
| Mod_files/IClampOU.mod        | NEURON mechanism: Ornstein-Uhlenbeck process for an input current| 
| Mod_files/Spikeout.mod        | NEURON Spike mechanism: reset the membrane potential after reaching a spike threshold. Also include a refractory time|
| Mod_files/xtra.mod            | NEURON mechanism: Pointer to set the extracellular field, i.e., the gradient of extracellular potential|

The mod files originates from the ModelDB database and the NEURON forum (see the comments in each file for  more details). They were slightly modified for the purpose of this project.

**The mod files need to be compiled in a specific order**, more precisely the fsin.mod mechanism needs to be compiled first.
Please use the provided shell script (compile_mod.sh) or reuse the same compilation order as in the script if you compile under Windows.

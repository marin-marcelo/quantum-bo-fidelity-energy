# quantum-bo-fidelity-energy


This is the repository of my research project ‘Observable-Guided Bayesian Optimisation for Quantum Circuits Fidelity and Ground State Energy Estimation’ for the MRes Machine Learning and Big Data in the Physical Sciences at Imperial College London.

The project deals with optimising the dynamics of quantum systems using Bayesian Optimisation. In particular, the same methodology is used to address problems in the field of Quantum Optimal Control (QOC) and Variational Quantum Eigensolver (VQE). For more information on this, please read the report.


The aim of the developed code is mainly investigative, so in order to have more control and to be able to make modifications easily, we opted for a low packaging of the different components of the algorithm. 

As mentioned in the report, several experiments were done in QOC, which can be re-done with the GHZ_Observables_BO.ipynb and GHZ_Statevector_BO.ipynb files. The same can be done with the VQE experiments, which were stored in the files H3_Observables_BO.ipynb and H4_Observables_BO.ipynb.


In the results folder, the results obtained when the experiments were performed in the university's HPC were stored.



The requirements are:

- qiskit
- gpytorch
- botorch
- SMT: Surrogate Modeling Toolbox
- matplotlib
- seaborn
# Copula-based-QCBM-estimation

- **generate_data.py**: Contains some utility fucntions to generate simulation data. In particular, a gaussian copula function to establish certain correlations between the marginals of the distribution, a binary encoding fucntion to translate classical data to the qubit space and histogram fucntion from a sequence of binary samples.
- **qcbm.py**: Class template for the QCBM generative model
- **train_hybrid_QCBM_random_data.ipynb**: Jupyter notebook for training/optimizing a QCBM that estimates the copula model for a given simulated data.
- **utils.py**: Contains utilities for the training of quantum circuits.

Possible improvements:

- Decouple target_probs variable from QCBM model initialization, should be set when training/fitting the model or evaluating the cost function. SPSA optimizer from qiskit does not support additional arguments like the scipy optimizers. Take a look at this: https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials/sample_vqe_program/qiskit_runtime_vqe_program.html

- Refactor the QCBM into a "QNN" architecture, similar to Pytorch.


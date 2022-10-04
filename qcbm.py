from collections import Counter
from itertools import product
from qiskit import *
from utils import KL
import numpy as np


class QCBM:

    def __init__(self,NUM_QUBITS_VAR,NUM_VARS,NUM_LAYERS,NUM_SHOTS,target_probs):
        self.NUM_QUBITS_VAR = NUM_QUBITS_VAR
        self.NUM_VARS = NUM_VARS
        self.NUM_LAYERS = NUM_LAYERS
        self.NUM_SHOTS = NUM_SHOTS
        self.target_probs = target_probs

    def GHZstate(self):
        qr = QuantumRegister(self.NUM_QUBITS_VAR*self.NUM_VARS)
        qc = QuantumCircuit(qr,name='GHZstate')

        for i in range(self.NUM_QUBITS_VAR):
            qc.h(i)
        for j in range(1,self.NUM_VARS):
            qc.cx(i,i+j*self.NUM_QUBITS_VAR)

        return qc.to_instruction()

    def parametrized_unitary(self,thetas,phis,rhos):

        """U_i transformation"""
        qr = QuantumRegister(self.NUM_QUBITS_VAR)
        qc = QuantumCircuit(qr,name = 'U_i')
        for i in range(self.NUM_QUBITS_VAR):
            qc.rz(thetas[i],i)
            qc.rx(phis[i],i)
        for i in range(self.NUM_QUBITS_VAR-1):
            qc.rzz(rhos[i],i,i+1)
        return qc.to_instruction()

    def construct_copula_ansatz(self,params):

        qr = QuantumRegister(self.NUM_QUBITS_VAR*self.NUM_VARS,'q')
        qc = QuantumCircuit(qr)
        qc.append(self.GHZstate(),qr[:])
        for l in range(self.NUM_LAYERS):
            thetas = params[0,:,l] # For RZ
            phis = params[1,:,l]   # For RX gates
            rhos = params[2,:,l] # For RZZ gates
            for k in range(0,self.NUM_VARS*self.NUM_QUBITS_VAR,self.NUM_QUBITS_VAR):
                qc.append(self.parametrized_unitary(thetas[k:k+self.NUM_QUBITS_VAR],
                                               phis[k:k+self.NUM_QUBITS_VAR],
                                               rhos[k:k+self.NUM_QUBITS_VAR]),
                                               qr[k:k+self.NUM_QUBITS_VAR])
        qc.measure_all()
        """
        cr = ClassicalRegister(self.NUM_VARS*self.NUM_QUBITS_VAR,'creg')
        qc.add_register(cr)
        for k in range(0,self.NUM_VARS*self.NUM_QUBITS_VAR,self.NUM_QUBITS_VAR):
            qc.measure(qr[k:k+self.NUM_QUBITS_VAR],cr[k:k+self.NUM_QUBITS_VAR][::-1])
        """
        return qc

    def get_born_probabilities(self,counts,num_shots):

        binary_states = sorted([x for x in map(''.join, product('01', repeat=self.NUM_QUBITS_VAR*self.NUM_VARS))])
        born_probs = np.zeros(2**(self.NUM_QUBITS_VAR*self.NUM_VARS))
        # if key is present, use relative freq of that particular binary state
        count_keys = sorted(counts.keys())
        for i in range(len(born_probs)):
            k = binary_states[i]
            if k in count_keys:
                born_probs[i] = counts.get(k) / num_shots
        #plot_histogram(counts)
        return np.asarray(born_probs)

    def estimate_distribution(self,params, num_shots = 2048):
        """
         This function performs a simulation with T shots and Q qbits. It's the parametrizable
         copula anstaz for learning a multivariate joint distribution in the copula space.
        Args:
            params: Flattened 3D array of form (2,NUM_QUBITS*NUM_VARS,NUM_LAYERS)
                    needed for the circuit.
            T: Integer number of shots that are going to take place in the simulation.
        Returns:
            born_probs: The born probabilities of the quantum state.
        """
        params = params.reshape(3,self.NUM_QUBITS_VAR*self.NUM_VARS,self.NUM_LAYERS)
        circuit = self.construct_copula_ansatz(params)
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(circuit, backend=simulator, shots=num_shots).result()
        counts = result.get_counts()
        born_probs = self.get_born_probabilities(counts,num_shots)
        return born_probs

    def sample(self,params,num_shots = 2048):

        """ Returns a numpy array of shape (NUM_SHOTS,NUM_VARS). Generates
        N_SHOTS pseudo-samples from the aproximated multivariate distribution"""
        params = params.reshape(3,self.NUM_QUBITS_VAR*self.NUM_VARS,self.NUM_LAYERS)
        circuit = self.construct_copula_ansatz(params)
        simulator = Aer.get_backend('qasm_simulator')
        measurements = execute(circuit.reverse_bits(), backend=simulator, shots=num_shots,memory=True).result()
        # Convert to pseudo-sample space
        U = []
        for bin_str in measurements.get_memory():
            pseudo_sample = []
            for i in range(0,self.NUM_QUBITS_VAR*self.NUM_VARS,self.NUM_QUBITS_VAR):
                pseudo_bin = bin_str[i:i+self.NUM_QUBITS_VAR][::-1]
                padding = (1/2**(self.NUM_QUBITS_VAR))*np.random.random_sample()
                pseudo_sample.append(int(pseudo_bin,2)*(1/2**(self.NUM_QUBITS_VAR))+padding)
            U.append(pseudo_sample)
        return np.asarray(U)

    def cost_function(self,params):

        born_probs = self.estimate_distribution(params,num_shots = self.NUM_SHOTS)
        loss = KL(born_probs,self.target_probs)
        return loss

from .envbase import EnvBase
from qiskit_aer import noise as noise_aer
import qiskit
import numpy as np

class CorrelatedNoise(EnvBase):
    def __init__(self,eta=0.1):
        super().__init__()
        self.n=5
        self.eta=eta
        self.set_noise()
    def set_noise(self):
        lam=0.002
        eta=self.eta
        px=eta * lam
        py=eta * lam
        pz=lam

        noise=noise_aer.pauli_error([('X',px),('Y',py),('Z',pz),('I',1-px-py-pz)])

        def xx_error(pxx=0.001):
            c=qiskit.QuantumCircuit(2)
            c.x(0)
            c.x(1)
            A=np.identity(2**2)*(1-pxx)**(1/2)
            B=qiskit.quantum_info.Operator(c).data*pxx**(1/2)
            return qiskit.quantum_info.Kraus([A,B])

        circuit = qiskit.QuantumCircuit(self.n)
        for i in range(self.n):
            circuit.append(noise, [i])
        circuit.append(xx_error(), [0, 1])
        circuit.append(xx_error(), [1, 2])
        circuit.append(xx_error(), [2, 3])
        circuit.append(xx_error(), [3, 4])
        circuit.append(xx_error(), [4, 0])

        kraus_operators = qiskit.quantum_info.Kraus(circuit).data
        self.noise_kraus=kraus_operators

class CorrelatedNoiseSingle(EnvBase):
    def __init__(self,eta=0.1):
        super().__init__()
        self.n=1
        self.eta=eta
        self.set_noise()
    def set_noise(self):
        lam=0.002
        eta=self.eta
        px=eta * lam
        py=eta * lam
        pz=lam

        noise=noise_aer.pauli_error([('X',px),('Y',py),('Z',pz),('I',1-px-py-pz)])

        def xx_error(pxx=0.001):
            c=qiskit.QuantumCircuit(1)
            c.x(0)
            A=np.identity(2**1)*(1-pxx)**(1/2)
            B=qiskit.quantum_info.Operator(c).data*pxx**(1/2)
            return qiskit.quantum_info.Kraus([A,B])

        circuit = qiskit.QuantumCircuit(self.n)
        for i in range(self.n):
            circuit.append(noise, [i])
        circuit.append(xx_error(), [0])

        kraus_operators = qiskit.quantum_info.Kraus(circuit).data
        self.noise_kraus=kraus_operators

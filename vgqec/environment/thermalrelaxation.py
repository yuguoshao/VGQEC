from .envbase import EnvBase
from qiskit_aer import noise as noise_aer
import qiskit
import numpy as np

class ThermalRelaxationNoise(EnvBase):
    def __init__(self,t=4):
        super().__init__()
        self.n=5
        self.t=t
        self.set_noise()

    def set_noise(self):
        t=self.t
        error1 = noise_aer.thermal_relaxation_error(97.51, 178.3, t)
        error2 = noise_aer.thermal_relaxation_error(127.61, 109.28, t)
        error3 = noise_aer.thermal_relaxation_error(92.68, 120.95, t)
        error4 = noise_aer.thermal_relaxation_error(79.36, 35.71, t)
        error5 = noise_aer.thermal_relaxation_error(19.76, 19.4, t)

        circuit = qiskit.QuantumCircuit(self.n)

        circuit.append(error1, [0])
        circuit.append(error2, [1])
        circuit.append(error3, [2])
        circuit.append(error4, [3])
        circuit.append(error5, [4])

        kraus_operators = qiskit.quantum_info.Kraus(circuit).data
        self.noise_kraus=kraus_operators

from .envbase import EnvBase
from qiskit_aer import noise as noise_aer
import qiskit

class AmplitudeDamping(EnvBase):
    def __init__(self,n,lam=0.1):
        super().__init__()
        self.n=n
        self.lam=lam

    def set_noise(self):
        circuit = qiskit.QuantumCircuit(self.n)
        error = noise_aer.amplitude_damping_error(self.lam)
        for i in range(self.n):
            circuit.append(error, [i])

        kraus_operators = qiskit.quantum_info.Kraus(circuit).data
        self.noise_kraus=kraus_operators

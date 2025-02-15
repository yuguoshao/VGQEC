import numpy as np
from . import VGQECCode,RepetitionCode
import qiskit
from qiskit.quantum_info import Statevector

class VGQEC_three_hybrid(VGQECCode):
    def __init__(self):
        super().__init__()
        self.n=3
        self.k=1
        self.basecode=RepetitionCode()
        self.init_gen()
    def update_encode_mat(self):
        #self.parameters
        # self.parameters
        par = self.parameters
        circuit0 = qiskit.QuantumCircuit(3)
        # circuit.x(0)
        circuit0.cx(0, 1)
        circuit0.cx(0, 2)

        circuit0.rz(par[0], 0)
        circuit0.rz(par[1], 1)
        circuit0.rz(par[2], 2)
        circuit0.rzz(par[3], 0, 1)
        circuit0.rzz(par[4], 1, 2)

        circuit0.rx(par[5], 1)
        circuit0.rzz(par[6], 0, 1)
        circuit0.rzz(par[7], 1, 2)

        circuit0.rx(par[8], 0)
        circuit0.rx(par[9], 1)
        circuit0.rx(par[10], 2)
        circuit0.rzz(par[11], 0, 1)
        circuit0.rzz(par[12], 1, 2)

        circuit0.rx(par[13], 1)
        circuit0.rzz(par[14], 0, 1)
        circuit0.rzz(par[15], 1, 2)
        circuit0.rx(par[16], 0)
        circuit0.rx(par[17], 2)

        circuit0.rz(par[18], 0)
        circuit0.rz(par[19], 1)
        circuit0.rz(par[20], 2)

        statevector0 = Statevector.from_instruction(circuit0)

        # 1
        circuit1 = qiskit.QuantumCircuit(5)
        circuit1.x(0)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)

        circuit1.rz(par[0], 0)
        circuit1.rz(par[1], 1)
        circuit1.rz(par[2], 2)
        circuit1.rzz(par[3], 0, 1)
        circuit1.rzz(par[4], 1, 2)

        circuit1.rx(par[5], 1)
        circuit1.rzz(par[6], 0, 1)
        circuit1.rzz(par[7], 1, 2)

        circuit1.rx(par[8], 0)
        circuit1.rx(par[9], 1)
        circuit1.rx(par[10], 2)
        circuit1.rzz(par[11], 0, 1)
        circuit1.rzz(par[12], 1, 2)

        circuit1.rx(par[13], 1)
        circuit1.rzz(par[14], 0, 1)
        circuit1.rzz(par[15], 1, 2)
        circuit1.rx(par[16], 0)
        circuit1.rx(par[17], 2)

        circuit1.rz(par[18], 0)
        circuit1.rz(par[19], 1)
        circuit1.rz(par[20], 2)
        statevector1 = Statevector.from_instruction(circuit1)

        self.encode_mat = np.array([statevector0.data, statevector1.data])
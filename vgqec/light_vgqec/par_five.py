import numpy as np
from . import VGQECCode,PerfectCode
import qiskit
from qiskit.quantum_info import Statevector

class VGQEC_five_light(VGQECCode):
    def __init__(self):
        super().__init__()
        self.n=5
        self.k=1
        self.num_para = 5
        self.basecode=PerfectCode()
        self.init_gen()
    def update_encode_mat(self):
        #self.parameters
        par=self.parameters
        circuit0 = qiskit.QuantumCircuit(5)
        # circuit.x(0)
        circuit0.cx(0, 1)
        circuit0.cx(0, 2)
        circuit0.cx(0, 3)
        circuit0.cx(0, 4)
        circuit0.h(0)
        circuit0.h(1)
        circuit0.h(2)
        circuit0.h(3)
        circuit0.h(4)
        circuit0.rzz(par[0], 0, 4)
        circuit0.rzz(par[1], 0, 1)
        circuit0.rzz(par[2], 1, 2)
        circuit0.rzz(par[3], 2, 3)
        circuit0.rzz(par[4], 3, 4)
        statevector0 = Statevector.from_instruction(circuit0)

        # 1
        circuit1 = qiskit.QuantumCircuit(5)
        circuit1.x(0)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.cx(0, 3)
        circuit1.cx(0, 4)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.h(2)
        circuit1.h(3)
        circuit1.h(4)
        circuit1.rzz(par[0], 0, 4)
        circuit1.rzz(par[1], 0, 1)
        circuit1.rzz(par[2], 1, 2)
        circuit1.rzz(par[3], 2, 3)
        circuit1.rzz(par[4], 3, 4)
        statevector1 = Statevector.from_instruction(circuit1)

        self.encode_mat=np.array([statevector0.data, statevector1.data])
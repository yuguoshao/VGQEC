import numpy as np
from . import VGQECCode,PerfectCode
import qiskit
from qiskit.quantum_info import Statevector

class VGQEC_five_hybrid(VGQECCode):
    def __init__(self):
        super().__init__()
        self.n=5
        self.k=1
        self.num_para = 60
        self.num_para_rec = 60
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

        circuit0.rz(par[5], 0)
        circuit0.rz(par[6], 1)
        circuit0.rz(par[7], 2)
        circuit0.rz(par[8], 3)
        circuit0.rz(par[9], 4)

        circuit0.rzz(par[10], 0, 1)
        circuit0.rzz(par[11], 3, 4)
        circuit0.rx(par[12], 1)
        circuit0.rx(par[13], 3)
        circuit0.rzz(par[14], 1, 2)
        circuit0.rzz(par[15], 2, 3)
        circuit0.rx(par[16], 2)

        circuit0.rzz(par[17], 1, 2)
        circuit0.rzz(par[18], 2, 3)
        circuit0.rx(par[19], 1)
        circuit0.rx(par[20], 3)
        circuit0.rzz(par[21], 0, 1)
        circuit0.rzz(par[22], 1, 2)
        circuit0.rzz(par[23], 2, 3)
        circuit0.rzz(par[24], 3, 4)

        circuit0.rx(par[25], 0)
        circuit0.rx(par[26], 2)
        circuit0.rx(par[27], 4)
        circuit0.rzz(par[28], 1, 2)
        circuit0.rzz(par[29], 2, 3)
        circuit0.rx(par[30], 1)
        circuit0.rx(par[31], 2)
        circuit0.rx(par[32], 3)
        circuit0.rzz(par[33], 1, 2)
        circuit0.rzz(par[34], 2, 3)
        circuit0.rx(par[35], 2)

        circuit0.rzz(par[36], 1, 2)
        circuit0.rzz(par[37], 2, 3)
        circuit0.rx(par[38], 1)
        circuit0.rx(par[39], 3)
        circuit0.rzz(par[40], 0, 1)
        circuit0.rzz(par[41], 3, 4)

        circuit0.rx(par[42], 1)
        circuit0.rx(par[43], 3)
        circuit0.rzz(par[44], 1, 2)
        circuit0.rzz(par[45], 2, 3)
        circuit0.rx(par[46], 2)
        circuit0.rzz(par[47], 1, 2)
        circuit0.rzz(par[48], 2, 3)
        circuit0.rx(par[49], 1)
        circuit0.rx(par[50], 3)
        circuit0.rzz(par[51], 0, 1)
        circuit0.rzz(par[52], 3, 4)
        circuit0.rx(par[53], 0)
        circuit0.rx(par[54], 4)

        circuit0.rz(par[55], 0)
        circuit0.rz(par[56], 1)
        circuit0.rz(par[57], 2)
        circuit0.rz(par[58], 3)
        circuit0.rz(par[59], 4)
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

        circuit1.rz(par[5], 0)
        circuit1.rz(par[6], 1)
        circuit1.rz(par[7], 2)
        circuit1.rz(par[8], 3)
        circuit1.rz(par[9], 4)

        circuit1.rzz(par[10], 0, 1)
        circuit1.rzz(par[11], 3, 4)
        circuit1.rx(par[12], 1)
        circuit1.rx(par[13], 3)
        circuit1.rzz(par[14], 1, 2)
        circuit1.rzz(par[15], 2, 3)
        circuit1.rx(par[16], 2)

        circuit1.rzz(par[17], 1, 2)
        circuit1.rzz(par[18], 2, 3)
        circuit1.rx(par[19], 1)
        circuit1.rx(par[20], 3)
        circuit1.rzz(par[21], 0, 1)
        circuit1.rzz(par[22], 1, 2)
        circuit1.rzz(par[23], 2, 3)
        circuit1.rzz(par[24], 3, 4)

        circuit1.rx(par[25], 0)
        circuit1.rx(par[26], 2)
        circuit1.rx(par[27], 4)
        circuit1.rzz(par[28], 1, 2)
        circuit1.rzz(par[29], 2, 3)
        circuit1.rx(par[30], 1)
        circuit1.rx(par[31], 2)
        circuit1.rx(par[32], 3)
        circuit1.rzz(par[33], 1, 2)
        circuit1.rzz(par[34], 2, 3)
        circuit1.rx(par[35], 2)

        circuit1.rzz(par[36], 1, 2)
        circuit1.rzz(par[37], 2, 3)
        circuit1.rx(par[38], 1)
        circuit1.rx(par[39], 3)
        circuit1.rzz(par[40], 0, 1)
        circuit1.rzz(par[41], 3, 4)

        circuit1.rx(par[42], 1)
        circuit1.rx(par[43], 3)
        circuit1.rzz(par[44], 1, 2)
        circuit1.rzz(par[45], 2, 3)
        circuit1.rx(par[46], 2)
        circuit1.rzz(par[47], 1, 2)
        circuit1.rzz(par[48], 2, 3)
        circuit1.rx(par[49], 1)
        circuit1.rx(par[50], 3)
        circuit1.rzz(par[51], 0, 1)
        circuit1.rzz(par[52], 3, 4)
        circuit1.rx(par[53], 0)
        circuit1.rx(par[54], 4)

        circuit1.rz(par[55], 0)
        circuit1.rz(par[56], 1)
        circuit1.rz(par[57], 2)
        circuit1.rz(par[58], 3)
        circuit1.rz(par[59], 4)
        statevector1 = Statevector.from_instruction(circuit1)

        self.encode_mat=np.array([statevector0.data, statevector1.data])
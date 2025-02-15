import numpy as np
from . import HybridScheme,PerfectCode
import qiskit
from qiskit.quantum_info import Statevector,Operator

class VGQEC_five_hybrid(HybridScheme):
    def __init__(self):
        super().__init__()
        self.n=5
        self.k=1
        self.num_para = 60
        self.num_para_rec = 7+35*3+14
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


    def gen_rec_kraus(self):
        # self.rec_parameters
        out=[]
        par = self.rec_parameters
        L = 3
        circuit = qiskit.QuantumCircuit(7)
        circuit.rz(par[0], 0)
        circuit.rz(par[1], 1)
        circuit.rz(par[2], 2)
        circuit.rz(par[3], 3)
        circuit.rz(par[4], 4)
        circuit.rz(par[5], 5)
        circuit.rz(par[6], 6)
        for i in range(L):
            ind=7+20*i
            circuit.rx(par[ind + 0], 0)
            circuit.rx(par[ind + 1], 1)
            circuit.rx(par[ind + 2], 2)
            circuit.rx(par[ind + 3], 3)
            circuit.rx(par[ind + 4], 4)
            circuit.rx(par[ind + 5], 5)
            circuit.rx(par[ind + 6], 6)

            circuit.rz(par[ind + 7], 0)
            circuit.rz(par[ind + 8], 1)
            circuit.rz(par[ind + 9], 2)
            circuit.rz(par[ind + 10], 3)
            circuit.rz(par[ind + 11], 4)
            circuit.rz(par[ind + 12], 5)
            circuit.rz(par[ind + 13], 6)


            circuit.rzz(par[ind + 14], 0, 1)
            circuit.rzz(par[ind + 15], 0, 2)
            circuit.rzz(par[ind + 16], 0, 3)
            circuit.rzz(par[ind + 17], 0, 4)
            circuit.rzz(par[ind + 18], 0, 5)
            circuit.rzz(par[ind + 19], 0, 6)

            circuit.rzz(par[ind + 20], 1, 2)
            circuit.rzz(par[ind + 21], 1, 3)
            circuit.rzz(par[ind + 22], 1, 4)
            circuit.rzz(par[ind + 23], 1, 5)
            circuit.rzz(par[ind + 24], 1, 6)

            circuit.rzz(par[ind + 25], 2, 3)
            circuit.rzz(par[ind + 26], 2, 4)
            circuit.rzz(par[ind + 27], 2, 5)
            circuit.rzz(par[ind + 28], 2, 6)
            circuit.rzz(par[ind + 29], 3, 4)
            circuit.rzz(par[ind + 30], 3, 5)
            circuit.rzz(par[ind + 31], 3, 6)
            circuit.rzz(par[ind + 32], 4, 5)
            circuit.rzz(par[ind + 33], 4, 6)
            circuit.rzz(par[ind + 34], 5, 6)

        ind=7+35*L
        circuit.rx(par[ind + 0], 0)
        circuit.rx(par[ind + 1], 1)
        circuit.rx(par[ind + 2], 2)
        circuit.rx(par[ind + 3], 3)
        circuit.rx(par[ind + 4], 4)
        circuit.rx(par[ind + 5], 5)
        circuit.rx(par[ind + 6], 6)

        circuit.rz(par[ind + 7], 0)
        circuit.rz(par[ind + 8], 1)
        circuit.rz(par[ind + 9], 2)
        circuit.rz(par[ind + 10], 3)
        circuit.rz(par[ind + 11], 4)
        circuit.rz(par[ind + 12], 5)
        circuit.rz(par[ind + 13], 6)


        unitary=Operator(circuit).data[:,:2**self.n]
        for i in range(2**(5-self.n)):
            ind=i
            ele=unitary[ind*2**self.n:(ind+1)*2**self.n]
            out.append(ele)
        self.rec_kraus=out
import numpy as np
from . import HybridScheme,PerfectCode
import qiskit,itertools
from qiskit.quantum_info import Statevector,Operator

class VGQEC_five_hybrid(HybridScheme):
    def __init__(self):
        super().__init__()
        self.n=5
        self.k=1
        self.num_para = 55
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

        circuit0.rzz(-0.5*np.pi, 0, 4)
        circuit0.rzz(-0.5*np.pi, 0, 1)
        circuit0.rzz(-0.5*np.pi, 1, 2)
        circuit0.rzz(-0.5*np.pi, 2, 3)
        circuit0.rzz(-0.5*np.pi, 3, 4)


        circuit0.rz(par[0], 0)
        circuit0.rz(par[1], 1)
        circuit0.rz(par[2], 2)
        circuit0.rz(par[3], 3)
        circuit0.rz(par[4], 4)

        circuit0.rzz(par[5], 0, 1)
        circuit0.rzz(par[6], 3, 4)
        circuit0.rx(par[7], 1)
        circuit0.rx(par[8], 3)
        circuit0.rzz(par[9], 1, 2)
        circuit0.rzz(par[10], 2, 3)
        circuit0.rx(par[11], 2)

        circuit0.rzz(par[12], 1, 2)
        circuit0.rzz(par[13], 2, 3)
        circuit0.rx(par[14], 1)
        circuit0.rx(par[15], 3)
        circuit0.rzz(par[16], 0, 1)
        circuit0.rzz(par[17], 1, 2)
        circuit0.rzz(par[18], 2, 3)
        circuit0.rzz(par[19], 3, 4)

        circuit0.rx(par[20], 0)
        circuit0.rx(par[21], 2)
        circuit0.rx(par[22], 4)
        circuit0.rzz(par[23], 1, 2)
        circuit0.rzz(par[24], 2, 3)
        circuit0.rx(par[25], 1)
        circuit0.rx(par[26], 2)
        circuit0.rx(par[27], 3)
        circuit0.rzz(par[28], 1, 2)
        circuit0.rzz(par[29], 2, 3)
        circuit0.rx(par[30], 2)

        circuit0.rzz(par[31], 1, 2)
        circuit0.rzz(par[32], 2, 3)
        circuit0.rx(par[33], 1)
        circuit0.rx(par[34], 3)
        circuit0.rzz(par[35], 0, 1)
        circuit0.rzz(par[36], 3, 4)

        circuit0.rx(par[37], 1)
        circuit0.rx(par[38], 3)
        circuit0.rzz(par[39], 1, 2)
        circuit0.rzz(par[40], 2, 3)
        circuit0.rx(par[41], 2)
        circuit0.rzz(par[42], 1, 2)
        circuit0.rzz(par[43], 2, 3)
        circuit0.rx(par[44], 1)
        circuit0.rx(par[45], 3)
        circuit0.rzz(par[46], 0, 1)
        circuit0.rzz(par[47], 3, 4)
        circuit0.rx(par[48], 0)
        circuit0.rx(par[49], 4)

        circuit0.rz(par[50], 0)
        circuit0.rz(par[51], 1)
        circuit0.rz(par[52], 2)
        circuit0.rz(par[53], 3)
        circuit0.rz(par[54], 4)
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

        circuit1.rzz(-0.5 * np.pi, 0, 4)
        circuit1.rzz(-0.5 * np.pi, 0, 1)
        circuit1.rzz(-0.5 * np.pi, 1, 2)
        circuit1.rzz(-0.5 * np.pi, 2, 3)
        circuit1.rzz(-0.5 * np.pi, 3, 4)


        circuit1.rz(par[0], 0)
        circuit1.rz(par[1], 1)
        circuit1.rz(par[2], 2)
        circuit1.rz(par[3], 3)
        circuit1.rz(par[4], 4)

        circuit1.rzz(par[5], 0, 1)
        circuit1.rzz(par[6], 3, 4)
        circuit1.rx(par[7], 1)
        circuit1.rx(par[8], 3)
        circuit1.rzz(par[9], 1, 2)
        circuit1.rzz(par[10], 2, 3)
        circuit1.rx(par[11], 2)

        circuit1.rzz(par[12], 1, 2)
        circuit1.rzz(par[13], 2, 3)
        circuit1.rx(par[14], 1)
        circuit1.rx(par[15], 3)
        circuit1.rzz(par[16], 0, 1)
        circuit1.rzz(par[17], 1, 2)
        circuit1.rzz(par[18], 2, 3)
        circuit1.rzz(par[19], 3, 4)

        circuit1.rx(par[20], 0)
        circuit1.rx(par[21], 2)
        circuit1.rx(par[22], 4)
        circuit1.rzz(par[23], 1, 2)
        circuit1.rzz(par[24], 2, 3)
        circuit1.rx(par[25], 1)
        circuit1.rx(par[26], 2)
        circuit1.rx(par[27], 3)
        circuit1.rzz(par[28], 1, 2)
        circuit1.rzz(par[29], 2, 3)
        circuit1.rx(par[30], 2)

        circuit1.rzz(par[31], 1, 2)
        circuit1.rzz(par[32], 2, 3)
        circuit1.rx(par[33], 1)
        circuit1.rx(par[34], 3)
        circuit1.rzz(par[35], 0, 1)
        circuit1.rzz(par[36], 3, 4)

        circuit1.rx(par[37], 1)
        circuit1.rx(par[38], 3)
        circuit1.rzz(par[39], 1, 2)
        circuit1.rzz(par[40], 2, 3)
        circuit1.rx(par[41], 2)
        circuit1.rzz(par[42], 1, 2)
        circuit1.rzz(par[43], 2, 3)
        circuit1.rx(par[44], 1)
        circuit1.rx(par[45], 3)
        circuit1.rzz(par[46], 0, 1)
        circuit1.rzz(par[47], 3, 4)
        circuit1.rx(par[48], 0)
        circuit1.rx(par[49], 4)

        circuit1.rz(par[50], 0)
        circuit1.rz(par[51], 1)
        circuit1.rz(par[52], 2)
        circuit1.rz(par[53], 3)
        circuit1.rz(par[54], 4)
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

            for (i,j) in enumerate(itertools.combinations(range(7),2)):
                circuit.rzz(par[ind + 14+i], j[0], j[1])

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
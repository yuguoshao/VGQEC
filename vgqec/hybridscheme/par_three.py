import numpy as np
from . import VGQECCode,RepetitionCode
import qiskit
from qiskit.quantum_info import Statevector,Operator

class VGQEC_three_hybrid(VGQECCode):
    def __init__(self):
        super().__init__()
        self.n=3
        self.k=1
        self.basecode=RepetitionCode()
        self.rec_parameters = None
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

        def gen_rec_kraus(self):
            # self.rec_parameters
            out=[]
            par = self.rec_parameters
            L = 3
            circuit = qiskit.QuantumCircuit(5)
            circuit.rz(par[0], 0)
            circuit.rz(par[1], 1)
            circuit.rz(par[2], 2)
            circuit.rz(par[3], 3)
            circuit.rz(par[4], 4)
            for i in range(L):
                ind=5+20*i
                circuit.rx(par[ind + 0], 0)
                circuit.rx(par[ind + 1], 1)
                circuit.rx(par[ind + 2], 2)
                circuit.rx(par[ind + 3], 3)
                circuit.rx(par[ind + 4], 4)
                circuit.rz(par[ind + 5], 0)
                circuit.rz(par[ind + 6], 1)
                circuit.rz(par[ind + 7], 2)
                circuit.rz(par[ind + 8], 3)
                circuit.rz(par[ind + 9], 4)

                circuit.rzz(par[ind + 10], 0, 1)
                circuit.rzz(par[ind + 11], 0, 2)
                circuit.rzz(par[ind + 12], 0, 3)
                circuit.rzz(par[ind + 13], 0, 4)
                circuit.rzz(par[ind + 14], 1, 2)
                circuit.rzz(par[ind + 15], 1, 3)
                circuit.rzz(par[ind + 16], 1, 4)
                circuit.rzz(par[ind + 17], 2, 3)
                circuit.rzz(par[ind + 18], 2, 4)
                circuit.rzz(par[ind + 19], 3, 4)

            ind=5+20*L
            circuit.rx(par[ind + 0], 0)
            circuit.rx(par[ind + 1], 1)
            circuit.rx(par[ind + 2], 2)
            circuit.rx(par[ind + 3], 3)
            circuit.rx(par[ind + 4], 4)
            circuit.rz(par[ind + 5], 0)
            circuit.rz(par[ind + 6], 1)
            circuit.rz(par[ind + 7], 2)
            circuit.rz(par[ind + 8], 3)
            circuit.rz(par[ind + 9], 4)

            unitary=Operator(circuit).data[:,:2**self.n]
            for i in range(2**(5-self.n)):
                ind=i
                ele=unitary[ind*2**self.n:(ind+1)*2**self.n]
                out.append(ele)
            self.rec_kraus=out




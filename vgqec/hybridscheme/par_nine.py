import pennylane as qml
from . import HybridScheme, SurfaceCode9
import itertools
import pennylane.numpy as np

def vec_matrix_fidelity(vec,matrix):
    #F(\rho_1, \rho_2) = Tr[\sqrt{\sqrt{\rho_1}\rho_2\sqrt{\rho_1}}]^2.
    innerproduct=np.dot(np.dot(vec.conjugate().reshape((1, len(vec))), matrix), vec)
    return np.real(innerproduct)


class VGQEC_nine_hybrid(HybridScheme):
    def __init__(self):
        super().__init__()
        self.n=9
        self.k=1
        self.num_para = 171
        self.num_para_rec = 9+(36+18)*3 +18
        self.basecode=SurfaceCode9()
        self.init_gen()
    def update_encode_mat(self):
        # self.parameters
        par = self.parameters

        if not hasattr(self, "_encode_qnode"):
            dev = qml.device("default.qubit", wires=self.n)

            def _circuit(state, par):
                qml.StatePrep(state, wires=range(self.n))

                qml.RZ(par[0], wires=0)
                qml.RZ(par[1], wires=1)
                qml.RZ(par[2], wires=2)
                qml.RZ(par[3], wires=3)
                qml.RZ(par[4], wires=4)
                qml.RZ(par[5], wires=5)
                qml.RZ(par[6], wires=6)
                qml.RZ(par[7], wires=7)
                qml.RZ(par[8], wires=8)

                qml.RX(par[9], wires=0)
                qml.RX(par[10], wires=4)
                qml.RX(par[11], wires=8)

                qml.IsingZZ(par[12], wires=[0, 1])
                qml.IsingZZ(par[13], wires=[3, 4])
                qml.IsingZZ(par[14], wires=[4, 5])
                qml.IsingZZ(par[15], wires=[7, 8])

                qml.RX(par[16], wires=0)
                qml.RX(par[17], wires=1)
                qml.RX(par[18], wires=3)
                qml.RX(par[19], wires=4)
                qml.RX(par[20], wires=5)
                qml.RX(par[21], wires=7)
                qml.RX(par[22], wires=8)

                qml.IsingZZ(par[23], wires=[0, 1])
                qml.IsingZZ(par[24], wires=[1, 2])
                qml.IsingZZ(par[25], wires=[3, 4])
                qml.IsingZZ(par[26], wires=[4, 5])
                qml.IsingZZ(par[27], wires=[6, 7])
                qml.IsingZZ(par[28], wires=[7, 8])

                qml.RX(par[29], wires=0)
                qml.RX(par[30], wires=1)
                qml.RX(par[31], wires=2)
                qml.RX(par[32], wires=4)
                qml.RX(par[33], wires=6)
                qml.RX(par[34], wires=7)
                qml.RX(par[35], wires=8)

                qml.IsingZZ(par[36], wires=[0, 1])
                qml.IsingZZ(par[37], wires=[1, 2])
                qml.IsingZZ(par[38], wires=[2, 3])
                qml.IsingZZ(par[39], wires=[5, 6])
                qml.IsingZZ(par[40], wires=[6, 7])
                qml.IsingZZ(par[41], wires=[7, 8])

                qml.RX(par[42], wires=0)
                qml.RX(par[43], wires=1)
                qml.RX(par[44], wires=2)
                qml.RX(par[45], wires=3)
                qml.RX(par[46], wires=5)
                qml.RX(par[47], wires=6)
                qml.RX(par[48], wires=7)
                qml.RX(par[49], wires=8)

                qml.IsingZZ(par[50], wires=[0, 1])
                qml.IsingZZ(par[51], wires=[1, 2])
                qml.IsingZZ(par[52], wires=[2, 3])
                qml.IsingZZ(par[53], wires=[3, 4])
                qml.IsingZZ(par[54], wires=[4, 5])
                qml.IsingZZ(par[55], wires=[5, 6])
                qml.IsingZZ(par[56], wires=[6, 7])
                qml.IsingZZ(par[57], wires=[7, 8])

                qml.RX(par[58], wires=0)
                qml.RX(par[59], wires=1)
                qml.RX(par[60], wires=2)
                qml.RX(par[61], wires=3)
                qml.RX(par[62], wires=4)
                qml.RX(par[63], wires=5)
                qml.RX(par[64], wires=6)
                qml.RX(par[65], wires=7)
                qml.RX(par[66], wires=8)

                qml.IsingZZ(par[67], wires=[0, 1])
                qml.IsingZZ(par[68], wires=[1, 2])
                qml.IsingZZ(par[69], wires=[2, 3])
                qml.IsingZZ(par[70], wires=[3, 4])
                qml.IsingZZ(par[71], wires=[4, 5])
                qml.IsingZZ(par[72], wires=[5, 6])
                qml.IsingZZ(par[73], wires=[6, 7])
                qml.IsingZZ(par[74], wires=[7, 8])

                qml.RX(par[75], wires=0)
                qml.RX(par[76], wires=1)
                qml.RX(par[77], wires=2)
                qml.RX(par[78], wires=3)
                qml.RX(par[79], wires=4)
                qml.RX(par[80], wires=5)
                qml.RX(par[81], wires=6)
                qml.RX(par[82], wires=7)
                qml.RX(par[83], wires=8)

                qml.IsingZZ(par[84], wires=[0, 1])
                qml.IsingZZ(par[85], wires=[1, 2])
                qml.IsingZZ(par[86], wires=[2, 3])
                qml.IsingZZ(par[87], wires=[3, 4])
                qml.IsingZZ(par[88], wires=[4, 5])
                qml.IsingZZ(par[89], wires=[5, 6])
                qml.IsingZZ(par[90], wires=[6, 7])
                qml.IsingZZ(par[91], wires=[7, 8])

                qml.RX(par[92], wires=1)
                qml.RX(par[93], wires=2)
                qml.RX(par[94], wires=3)
                qml.RX(par[95], wires=4)
                qml.RX(par[96], wires=5)
                qml.RX(par[97], wires=6)
                qml.RX(par[98], wires=7)

                qml.IsingZZ(par[99], wires=[0, 1])
                qml.IsingZZ(par[100], wires=[1, 2])
                qml.IsingZZ(par[101], wires=[2, 3])
                qml.IsingZZ(par[102], wires=[3, 4])
                qml.IsingZZ(par[103], wires=[4, 5])
                qml.IsingZZ(par[104], wires=[5, 6])
                qml.IsingZZ(par[105], wires=[6, 7])
                qml.IsingZZ(par[106], wires=[7, 8])

                qml.RX(par[107], wires=1)
                qml.RX(par[108], wires=2)
                qml.RX(par[109], wires=3)
                qml.RX(par[110], wires=4)
                qml.RX(par[111], wires=5)
                qml.RX(par[112], wires=6)
                qml.RX(par[113], wires=7)

                qml.IsingZZ(par[114], wires=[0, 1])
                qml.IsingZZ(par[115], wires=[1, 2])
                qml.IsingZZ(par[116], wires=[2, 3])
                qml.IsingZZ(par[117], wires=[3, 4])
                qml.IsingZZ(par[118], wires=[4, 5])
                qml.IsingZZ(par[119], wires=[5, 6])
                qml.IsingZZ(par[120], wires=[6, 7])
                qml.IsingZZ(par[121], wires=[7, 8])

                qml.RX(par[122], wires=0)
                qml.RX(par[123], wires=1)
                qml.RX(par[124], wires=2)
                qml.RX(par[125], wires=3)
                qml.RX(par[126], wires=4)
                qml.RX(par[127], wires=5)
                qml.RX(par[128], wires=6)
                qml.RX(par[129], wires=7)
                qml.RX(par[130], wires=8)

                qml.IsingZZ(par[131], wires=[0, 1])
                qml.IsingZZ(par[132], wires=[1, 2])
                qml.IsingZZ(par[133], wires=[2, 3])
                qml.IsingZZ(par[134], wires=[3, 4])
                qml.IsingZZ(par[135], wires=[4, 5])
                qml.IsingZZ(par[136], wires=[5, 6])
                qml.IsingZZ(par[137], wires=[6, 7])
                qml.IsingZZ(par[138], wires=[7, 8])

                qml.RX(par[139], wires=1)
                qml.RX(par[140], wires=2)
                qml.RX(par[141], wires=3)
                qml.RX(par[142], wires=4)
                qml.RX(par[143], wires=5)
                qml.RX(par[144], wires=6)
                qml.RX(par[145], wires=7)

                qml.IsingZZ(par[146], wires=[1, 2])
                qml.IsingZZ(par[147], wires=[2, 3])
                qml.IsingZZ(par[148], wires=[3, 4])
                qml.IsingZZ(par[149], wires=[4, 5])
                qml.IsingZZ(par[150], wires=[5, 6])
                qml.IsingZZ(par[151], wires=[6, 7])

                qml.RX(par[152], wires=2)
                qml.RX(par[153], wires=3)
                qml.RX(par[154], wires=5)
                qml.RX(par[155], wires=6)

                qml.IsingZZ(par[156], wires=[2, 3])
                qml.IsingZZ(par[157], wires=[5, 6])

                qml.RX(par[158], wires=3)
                qml.RX(par[159], wires=5)

                qml.IsingZZ(par[160], wires=[3, 4])
                qml.IsingZZ(par[161], wires=[4, 5])

                qml.RZ(par[162], wires=0)
                qml.RZ(par[163], wires=1)
                qml.RZ(par[164], wires=2)
                qml.RZ(par[165], wires=3)
                qml.RZ(par[166], wires=4)
                qml.RZ(par[167], wires=5)
                qml.RZ(par[168], wires=6)
                qml.RZ(par[169], wires=7)
                qml.RZ(par[170], wires=8)


                return qml.state()

            self._encode_qnode = qml.QNode(_circuit, dev, interface="auto", diff_method="best")

        state_list = [
            self._encode_qnode(state_ele, par) for state_ele in self.basecode.encode_mat
        ]
        self.encode_mat = qml.math.stack(state_list, axis=0)



    def gen_rec_kraus(self):
        # self.rec_parameters
        out=[]
        par = self.rec_parameters
        L = 3

        if not hasattr(self, "_rec_unitary_fn"):
            #dev = qml.device("default.qubit", wires=self.n)

            def _rec_circuit(par):
                #rho_full = rho
                #qml.QubitDensityMatrix(rho_full, wires=range(self.n))
                qml.RZ(par[0], wires=0)
                qml.RZ(par[1], wires=1)
                qml.RZ(par[2], wires=2)
                qml.RZ(par[3], wires=3)
                qml.RZ(par[4], wires=4)
                qml.RZ(par[5], wires=5)
                qml.RZ(par[6], wires=6)
                qml.RZ(par[7], wires=7)
                qml.RZ(par[8], wires=8)

                for i in range(L):
                    ind = 9 + (36 + 18) * i
                    qml.RX(par[ind + 0], wires=0)
                    qml.RX(par[ind + 1], wires=1)
                    qml.RX(par[ind + 2], wires=2)
                    qml.RX(par[ind + 3], wires=3)
                    qml.RX(par[ind + 4], wires=4)
                    qml.RX(par[ind + 5], wires=5)
                    qml.RX(par[ind + 6], wires=6)
                    qml.RX(par[ind + 7], wires=7)
                    qml.RX(par[ind + 8], wires=8)

                    qml.RZ(par[ind + 9], wires=0)
                    qml.RZ(par[ind + 10], wires=1)
                    qml.RZ(par[ind + 11], wires=2)
                    qml.RZ(par[ind + 12], wires=3)
                    qml.RZ(par[ind + 13], wires=4)
                    qml.RZ(par[ind + 14], wires=5)
                    qml.RZ(par[ind + 15], wires=6)
                    qml.RZ(par[ind + 16], wires=7)
                    qml.RZ(par[ind + 17], wires=8)

                    for offset, (j, k) in enumerate(itertools.combinations(range(9), 2)):
                        qml.IsingZZ(par[ind + 18 + offset], wires=[j, k])

                ind = 9 + (36 + 18) * L
                qml.RX(par[ind + 0], wires=0)
                qml.RX(par[ind + 1], wires=1)
                qml.RX(par[ind + 2], wires=2)
                qml.RX(par[ind + 3], wires=3)
                qml.RX(par[ind + 4], wires=4)
                qml.RX(par[ind + 5], wires=5)
                qml.RX(par[ind + 6], wires=6)
                qml.RX(par[ind + 7], wires=7)
                qml.RX(par[ind + 8], wires=8)

                qml.RZ(par[ind + 9], wires=0)
                qml.RZ(par[ind + 10], wires=1)
                qml.RZ(par[ind + 11], wires=2)
                qml.RZ(par[ind + 12], wires=3)
                qml.RZ(par[ind + 13], wires=4)
                qml.RZ(par[ind + 14], wires=5)
                qml.RZ(par[ind + 15], wires=6)
                qml.RZ(par[ind + 16], wires=7)
                qml.RZ(par[ind + 17], wires=8)


            #self._rec_unitary_fn = qml.QNode(_rec_circuit, dev, interface="auto", diff_method="backprop")
            #self._decode_qnode = qml.QNode(_rec_circuit, dev, interface="auto", diff_method="backprop")
            self._rec_unitary_fn = qml.matrix(_rec_circuit, wire_order=range(self.n))

        unitary = self._rec_unitary_fn(par)
        #cols = 2 ** self.n
        #unitary = unitary[:, :cols]
        #for i in range(2 ** (11 - self.n)):
        #    ele = unitary[i * cols:(i + 1) * cols]
        #    out.append(ele)
        #self.rec_kraus = lambda rho : self._rec_unitary_fn(rho, par)
        self.rec_kraus = [unitary]

    def encode(self,logical_state):
        if 2**self.k!=len(logical_state):
            raise ValueError('logical_state length must be 2^k')
        return np.dot(self.encode_mat.T, logical_state)

    def decode(self,density_matrix):
        out=np.zeros((2**self.n,2**self.n),dtype=np.complex128)
        for ele in self.rec_kraus:
            out+=ele@density_matrix@np.conjugate(ele.T)
        return self.base_decode(out)

    def base_decode(self,density_matrix):
        out=np.zeros((2**self.k,2**self.k),dtype=np.complex128)
        for ele in self.basecode.rec_kraus:
            out+=ele@density_matrix@ele.T.conjugate()
        return out

    def train_set_fidelity(self,density_matrix_set,ave=False):
        res= [vec_matrix_fidelity(state,density_matrix) for state,density_matrix in zip(self.train_set,density_matrix_set)]
        if ave:
            sum = np.tensor([0.0])
            for ele in res:
                sum += ele
            return sum/len(res)
        else:
            return res



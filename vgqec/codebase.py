import numpy as np


def vec_matrix_fidelity(vec,matrix):
    #F(\rho_1, \rho_2) = Tr[\sqrt{\sqrt{\rho_1}\rho_2\sqrt{\rho_1}}]^2.
    innerproduct=np.dot(np.dot(vec.conjugate().reshape((1, len(vec))), matrix), vec)
    return innerproduct


class CodeBase:
    def __init__(self):
        self.k = None
        self.n = None
        self.rec_kraus = None
        self.train_set = None


    def init_gen(self):
        self.gen_rec_kraus()
        self.get_train_set()

    def encode(self,logical_state):
        if 2**self.k!=len(logical_state):
            raise ValueError('logical_state length must be 2^k')
        return np.dot(self.encode_mat.T, logical_state)
    def encode_density_matrix(self,density_matrix):
        if 2**self.k!=density_matrix.shape[0]:
            raise ValueError('density_matrix size must be 2^k')
        return np.dot(self.encode_mat.T, density_matrix).dot(self.encode_mat.conjugate())

    def decode(self,density_matrix):
        out=np.zeros((2**self.k,2**self.k),dtype=np.complex128)
        for ele in self.rec_kraus:
            out+=ele@density_matrix@ele.T.conjugate()
        return out

    def get_train_set(self):
        # Generate k-qubit 2-design state
        if self.k==1:
            _random_state = []
            _random_state.append(np.array([1, 0]))
            _random_state.append(np.array([1 / np.sqrt(3), np.sqrt(2 / 3)]))
            _random_state.append(np.array([1 / np.sqrt(3), np.sqrt(2 / 3) * np.exp(2 / 3 * np.pi * 1j)]))
            _random_state.append(np.array([1 / np.sqrt(3), np.sqrt(2 / 3) * np.exp(4 / 3 * np.pi * 1j)]))
            self.train_set = _random_state
        else:
            raise NotImplementedError('k>1 is not implemented yet')

    def encoded_train_set(self):
        return [self.encode(state) for state in self.train_set]

    def train_set_fidelity(self,density_matrix_set,ave=False):
        res= [vec_matrix_fidelity(state,density_matrix) for state,density_matrix in zip(self.train_set,density_matrix_set)]
        if ave:
            return np.mean(res)
        else:
            return res

    def gen_rec_kraus(self):
        raise NotImplementedError('gen_rec_kraus is not implemented yet')

    def check_kraus(self):
        # sum ele.T.conjugate()@ele for all ele in rec_kraus should be identity matrix
        sum=np.zeros((2**self.n,2**self.n),dtype=np.complex128)
        for ele in self.rec_kraus:
            sum+=ele.T.conjugate()@ele
        return np.linalg.norm(sum-np.identity(2**self.n))

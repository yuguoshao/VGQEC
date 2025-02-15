import numpy as np
import cvxpy


def vec2mat(vec):
    # state vector to density matrix
    return np.outer(vec, vec.conj())

class EnvBase:
    def __init__(self):
        self.n=None
        self.noise_kraus=None

    def set_noise(self):
        raise NotImplementedError('set_noise is not implemented yet')


    def optimal_fidelity(self,code):
        vec=code.encode_mat
        k=code.k
        n=code.n
        if self.n!=n:
            raise ValueError('n is not matched')
        right = vec.reshape(1, 2 ** (k + n))
        encode = np.dot(right.conj().T, right)

        encode_4d = encode.reshape(2 ** k, 2 ** n, 2 ** k, 2 ** n)
        sums = np.zeros((2 ** k, 2 ** n, 2 ** k, 2 ** n), dtype=np.complex128)
        # sums2=np.zeros((2**(self.n+self.k),2**(self.k+self.n)), dtype=np.complex128)
        for E in self.noise_kraus:
            sums += np.einsum('abcd,de->abce', np.einsum('eb,abcd->aecd', E, encode_4d), E.conj().T)

        encode_choi=sums
        encode_choi_4d = np.einsum('abcd->dcba', encode_choi)
        encode_choi_2d = encode_choi_4d.reshape(2 ** (n + k), 2 ** (n + k))
        X = cvxpy.Variable((2 ** (n + k), 2 ** (n + k)), hermitian=True)
        constraints = [X >> 0, cvxpy.partial_trace(X, (2 ** n, 2 ** k), axis=1) == np.eye(2 ** self.n)]
        prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.real(cvxpy.trace(X @ encode_choi_2d))), constraints)
        prob.solve(eps=1e-9, solver=cvxpy.SCS)
        return prob.value/2**(2*k)

    def evo_mat(self,density_matrix):
        res=np.zeros_like(density_matrix,dtype=np.complex128)
        for ele in self.noise_kraus:
            res+=ele@density_matrix@ele.T.conjugate()
        return res

    def evo_density_set(self,set):
        return [self.evo_mat(density_matrix) for density_matrix in set]

    def evo_vec_set(self,set):
        density_matrix_set=[vec2mat(vec) for vec in set]
        return self.evo_density_set(density_matrix_set)


    def state_fidelity(self,code,ave=False):
        vec_set=code.encoded_train_set()
        evoed_set=self.evo_vec_set(vec_set)
        decoded_set=[code.decode(ele) for ele in evoed_set]
        res=code.train_set_fidelity(decoded_set,ave)
        return res

    def channel_fidelity(self,code):
        # Can only handle k=1
        b00=np.array([[1,0],[0,0]])
        b01=np.array([[0,1],[0,0]])
        b10=np.array([[0,0],[1,0]])
        b11=np.array([[0,0],[0,1]])

        density_set=[code.encode_density_matrix(ele) for ele in [b00,b01,b10,b11]]
        evoed_set=self.evo_density_set(density_set)
        decoded_set=[code.decode(ele) for ele in evoed_set]
        fid=decoded_set[0][0,0]+decoded_set[1][0,1]+decoded_set[2][1,0]+decoded_set[3][1,1]
        fid=fid/2**2
        return fid.real




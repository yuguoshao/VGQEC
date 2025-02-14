import numpy as np
from . import CodeBase

I=np.array([[1,0],[0,1]])
X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
Z=np.array([[1,0],[0,-1]])

pauli_dict={0:I,1:X,2:Y,3:Z}

class PerfectCode(CodeBase):
    def __init__(self):
        super().__init__()
        self.n = 5
        self.k = 1
        self.encode_mat = np.array([
            [1,-1,-1,-1,-1,1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1,1,-1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1],
            [1, 1, 1,-1, 1,1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1,1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1]])/np.sqrt(32)
        self.init_gen()

    def encode(self, logical_state):
        super().encode(logical_state)
        return np.dot(self.encode_mat.T, logical_state)

    def gen_rec_kraus(self):
        core=self.encode_mat
        res=[core]
        for i in range(15):
            indexs=i//3
            type=i%3+1
            temp=pauli_dict[type]
            for j in range(1,5):
                if j==indexs:
                    temp=np.kron(temp,pauli_dict[type])
                else:
                    temp=np.kron(temp,I)

            res.append(np.dot(core,temp))
        self.rec_kraus=res

import numpy as np
from . import CodeBase

class RepetitionCode(CodeBase):
    def __init__(self):
        super().__init__()
        self.n=3
        self.k=1
        self.encode_mat=np.array([[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1]])
        self.init_gen()
        self.gen_rec_kraus()


    def gen_rec_kraus(self):
        num=3
        n = 2 ** num
        krauses = []
        for i in range(int(n / 2)):
            kraus = np.zeros([n, n])
            if bin(i).count('1') < num / 2:
                kraus[0][i] = 1
                kraus[n - 1][n - i - 1] = 1
            else:
                kraus[n - 1][i] = 1
                kraus[0][n - i - 1] = 1
            krauses.append(kraus)
        self.rec_kraus= krauses

class RepetitionCodeFive(CodeBase):
    def __init__(self):
        super().__init__()
        self.n=5
        self.k=1
        self.encode_mat=np.array([[ 0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,
         0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,
         0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,
         0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,
         0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,
         0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,
         0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,
         0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j],
       [ 0.1767767+0.j, -0.1767767+0.j, -0.1767767+0.j,  0.1767767+0.j,
        -0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j, -0.1767767+0.j,
        -0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j, -0.1767767+0.j,
         0.1767767+0.j, -0.1767767+0.j, -0.1767767+0.j,  0.1767767+0.j,
        -0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j, -0.1767767+0.j,
         0.1767767+0.j, -0.1767767+0.j, -0.1767767+0.j,  0.1767767+0.j,
         0.1767767+0.j, -0.1767767+0.j, -0.1767767+0.j,  0.1767767+0.j,
        -0.1767767+0.j,  0.1767767+0.j,  0.1767767+0.j, -0.1767767+0.j]])
        self.init_gen()


    def gen_rec_kraus(self):
        pass
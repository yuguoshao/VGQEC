import numpy as np
from . import CodeBase

class NoProtection(CodeBase):
    def __init__(self):
        super().__init__()
        self.n=1
        self.k=1
        self.encode_mat=np.array([[1,0],[0,1]])
        self.init_gen()
        self.gen_rec_kraus()


    def gen_rec_kraus(self):
        krauses = [np.array([[1,0],[0,1]])]
        return krauses
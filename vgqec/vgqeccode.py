import numpy as np
import codebase
CodeBase=codebase.CodeBase

class VGQECCode(CodeBase):
    def __init__(self):
        super().__init__()
        self.basecode=None
        self.encode_mat=None
        self.parameters=None

    def init_gen(self):
        super().init_gen()
        #self.rec_kraus=self.basecode.rec_kraus


    def update_encode_mat(self):
        raise NotImplementedError('update_encode_mat is not implemented yet')

    def set_parameters(self,para):
        self.parameters=para
        self.update_encode_mat()
    def decode(self,density_matrix):
        out=np.zeros((2**self.n,2**self.n),dtype=np.complex128)
        for ele in self.rec_kraus:
            out+=ele@density_matrix@ele.T.conjugate()
        return self.basecode.decode(out)


    def gen_rec_kraus(self):
        pass
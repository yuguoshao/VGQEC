import numpy as np
import codebase
CodeBase=codebase.CodeBase

class VGQECCode(CodeBase):
    def __init__(self):
        super().__init__()
        self.basecode=None
        self.encode_mat=None
        self.parameters=None

    def encode(self, logical_state):
        super().encode(logical_state)
        return np.dot(self.encode_mat.T, logical_state)

    def update_encode_mat(self):
        raise NotImplementedError('update_encode_mat is not implemented yet')

    def set_parameters(self,para):
        self.parameters=para
        self.update_encode_mat()

    @property
    def rec_kraus(self):
        return self.basecode.rec_kraus

    def gen_rec_kraus(self):
        pass
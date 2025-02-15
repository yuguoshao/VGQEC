import numpy as np
from . import VGQECCode,RepetitionCode

class VGQEC_three_light(VGQECCode):
    def __init__(self):
        super().__init__()
        self.n=3
        self.k=1
        self.num_para=1
        self.basecode=RepetitionCode()
        self.init_gen()
    def update_encode_mat(self):
        #self.parameters
        alpha=self.parameters[0]
        a=1+np.exp(1j*alpha)
        a=a/2
        b=1-np.exp(1j*alpha)
        b=b/2
        vec=np.array([[a,0,0,b,0,0,0,0],[0,0,0,0,b,0,0,a]])/np.linalg.norm([a,b])
        self.encode_mat=vec
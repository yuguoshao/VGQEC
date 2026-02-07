import numpy as np
from . import HybridScheme, SurfaceCode9
import qiskit,itertools
from qiskit.quantum_info import Statevector,Operator

class VGQEC_nine_hybrid(HybridScheme):
    def __init__(self):
        super().__init__()
        self.n=9
        self.k=1
        self.num_para = 171
        self.num_para_rec = 11+(36+22)*3 +22
        self.basecode=SurfaceCode9()
        self.init_gen()
    def update_encode_mat(self):
        #self.parameters
        par=self.parameters
        circuit0 = qiskit.QuantumCircuit(9)
        state_list = []
        for i in range(2):
            circuit0.prepare_state(self.basecode.encode_mat[i])

            circuit0.rz(par[0], 0)
            circuit0.rz(par[1], 1)
            circuit0.rz(par[2], 2)
            circuit0.rz(par[3], 3)
            circuit0.rz(par[4], 4)
            circuit0.rz(par[5], 5)
            circuit0.rz(par[6], 6)
            circuit0.rz(par[7], 7)
            circuit0.rz(par[8], 8)

            circuit0.rx(par[9],0)
            circuit0.rx(par[10],4)
            circuit0.rx(par[11],8)

            circuit0.rzz(par[12],0,1)
            circuit0.rzz(par[13],3,4)
            circuit0.rzz(par[14],4,5)
            circuit0.rzz(par[15],7,8)

            circuit0.rx(par[16],0)
            circuit0.rx(par[17],1)
            circuit0.rx(par[18],3)
            circuit0.rx(par[19],4)
            circuit0.rx(par[20],5)
            circuit0.rx(par[21],7)
            circuit0.rx(par[22],8)

            circuit0.rzz(par[23],0,1)
            circuit0.rzz(par[24],1,2)
            circuit0.rzz(par[25],3,4)
            circuit0.rzz(par[26],4,5)
            circuit0.rzz(par[27],6,7)
            circuit0.rzz(par[28],7,8)

            circuit0.rx(par[29],0)
            circuit0.rx(par[30],1)
            circuit0.rx(par[31],2)
            circuit0.rx(par[32],4)
            circuit0.rx(par[33],6)
            circuit0.rx(par[34],7)
            circuit0.rx(par[35],8)

            circuit0.rzz(par[36],0,1)
            circuit0.rzz(par[37],1,2)
            circuit0.rzz(par[38],2,3)
            circuit0.rzz(par[39],5,6)
            circuit0.rzz(par[40],6,7)
            circuit0.rzz(par[41],7,8)

            circuit0.rx(par[42],0)
            circuit0.rx(par[43],1)
            circuit0.rx(par[44],2)
            circuit0.rx(par[45],3)
            circuit0.rx(par[46],5)
            circuit0.rx(par[47],6)
            circuit0.rx(par[48],7)
            circuit0.rx(par[49],8)

            circuit0.rzz(par[50],0,1)
            circuit0.rzz(par[51],1,2)
            circuit0.rzz(par[52],2,3)
            circuit0.rzz(par[53],3,4)
            circuit0.rzz(par[54],4,5)
            circuit0.rzz(par[55],5,6)
            circuit0.rzz(par[56],6,7)
            circuit0.rzz(par[57],7,8)

            circuit0.rx(par[58],0)
            circuit0.rx(par[59],1)
            circuit0.rx(par[60],2)
            circuit0.rx(par[61],3)
            circuit0.rx(par[62],4)
            circuit0.rx(par[63],5)
            circuit0.rx(par[64],6)
            circuit0.rx(par[65],7)
            circuit0.rx(par[66],8)

            circuit0.rzz(par[67],0,1)
            circuit0.rzz(par[68],1,2)
            circuit0.rzz(par[69],2,3)
            circuit0.rzz(par[70],3,4)
            circuit0.rzz(par[71],4,5)
            circuit0.rzz(par[72],5,6)
            circuit0.rzz(par[73],6,7)
            circuit0.rzz(par[74],7,8)

            circuit0.rx(par[75],0)
            circuit0.rx(par[76],1)
            circuit0.rx(par[77],2)
            circuit0.rx(par[78],3)
            circuit0.rx(par[79],4)
            circuit0.rx(par[80],5)
            circuit0.rx(par[81],6)
            circuit0.rx(par[82],7)
            circuit0.rx(par[83],8)

            circuit0.rzz(par[84],0,1)
            circuit0.rzz(par[85],1,2)
            circuit0.rzz(par[86],2,3)
            circuit0.rzz(par[87],3,4)
            circuit0.rzz(par[88],4,5)
            circuit0.rzz(par[89],5,6)
            circuit0.rzz(par[90],6,7)
            circuit0.rzz(par[91],7,8)

            circuit0.rx(par[92],1)
            circuit0.rx(par[93],2)
            circuit0.rx(par[94],3)
            circuit0.rx(par[95],4)
            circuit0.rx(par[96],5)
            circuit0.rx(par[97],6)
            circuit0.rx(par[98],7)

            circuit0.rzz(par[99],0,1)
            circuit0.rzz(par[100],1,2)
            circuit0.rzz(par[101],2,3)
            circuit0.rzz(par[102],3,4)
            circuit0.rzz(par[103],4,5)
            circuit0.rzz(par[104],5,6)
            circuit0.rzz(par[105],6,7)
            circuit0.rzz(par[106],7,8)

            circuit0.rx(par[107],1)
            circuit0.rx(par[108],2)
            circuit0.rx(par[109],3)
            circuit0.rx(par[110],4)
            circuit0.rx(par[111],5)
            circuit0.rx(par[112],6)
            circuit0.rx(par[113],7)

            circuit0.rzz(par[114],0,1)
            circuit0.rzz(par[115],1,2)
            circuit0.rzz(par[116],2,3)
            circuit0.rzz(par[117],3,4)
            circuit0.rzz(par[118],4,5)
            circuit0.rzz(par[119],5,6)
            circuit0.rzz(par[120],6,7)
            circuit0.rzz(par[121],7,8)

            circuit0.rx(par[122],0)
            circuit0.rx(par[123],1)
            circuit0.rx(par[124],2)
            circuit0.rx(par[125],3)
            circuit0.rx(par[126],4)
            circuit0.rx(par[127],5)
            circuit0.rx(par[128],6)
            circuit0.rx(par[129],7)
            circuit0.rx(par[130],8)

            circuit0.rzz(par[131],0,1)
            circuit0.rzz(par[132],1,2)
            circuit0.rzz(par[133],2,3)
            circuit0.rzz(par[134],3,4)
            circuit0.rzz(par[135],4,5)
            circuit0.rzz(par[136],5,6)
            circuit0.rzz(par[137],6,7)
            circuit0.rzz(par[138],7,8)

            circuit0.rx(par[139],1)
            circuit0.rx(par[140],2)
            circuit0.rx(par[141],3)
            circuit0.rx(par[142],4)
            circuit0.rx(par[143],5)
            circuit0.rx(par[144],6)
            circuit0.rx(par[145],7)

            circuit0.rzz(par[146],1,2)
            circuit0.rzz(par[147],2,3)
            circuit0.rzz(par[148],3,4)
            circuit0.rzz(par[149],4,5)
            circuit0.rzz(par[150],5,6)
            circuit0.rzz(par[151],6,7)

            circuit0.rx(par[152],2)
            circuit0.rx(par[153],3)
            circuit0.rx(par[154],5)
            circuit0.rx(par[155],6)

            circuit0.rzz(par[156],2,3)
            circuit0.rzz(par[157],5,6)

            circuit0.rx(par[158],3)
            circuit0.rx(par[159],5)

            circuit0.rzz(par[160],3,4)
            circuit0.rzz(par[161],4,5)


            circuit0.rz(par[162], 0)
            circuit0.rz(par[163], 1)
            circuit0.rz(par[164], 2)
            circuit0.rz(par[165], 3)
            circuit0.rz(par[166], 4)
            circuit0.rz(par[167], 5)
            circuit0.rz(par[168], 6)
            circuit0.rz(par[169], 7)
            circuit0.rz(par[170], 8)

            state = Statevector.from_instruction(circuit0)

            state_list.append(state.data)

        self.encode_mat=np.array(state_list)



    def gen_rec_kraus(self):
        # self.rec_parameters
        out=[]
        par = self.rec_parameters
        L = 3
        circuit = qiskit.QuantumCircuit(11)
        circuit.rz(par[0], 0)
        circuit.rz(par[1], 1)
        circuit.rz(par[2], 2)
        circuit.rz(par[3], 3)
        circuit.rz(par[4], 4)
        circuit.rz(par[5], 5)
        circuit.rz(par[6], 6)
        circuit.rz(par[7], 7)
        circuit.rz(par[8], 8)
        circuit.rz(par[9], 9)
        circuit.rz(par[10], 10)
        for i in range(L):
            ind=11+(36+22)*i
            circuit.rx(par[ind + 0], 0)
            circuit.rx(par[ind + 1], 1)
            circuit.rx(par[ind + 2], 2)
            circuit.rx(par[ind + 3], 3)
            circuit.rx(par[ind + 4], 4)
            circuit.rx(par[ind + 5], 5)
            circuit.rx(par[ind + 6], 6)
            circuit.rx(par[ind + 7], 7)
            circuit.rx(par[ind + 8], 8)
            circuit.rx(par[ind + 9], 9)
            circuit.rx(par[ind + 10], 10)

            circuit.rz(par[ind + 11], 0)
            circuit.rz(par[ind + 12], 1)
            circuit.rz(par[ind + 13], 2)
            circuit.rz(par[ind + 14], 3)
            circuit.rz(par[ind + 15], 4)
            circuit.rz(par[ind + 16], 5)
            circuit.rz(par[ind + 17], 6)
            circuit.rz(par[ind + 18], 7)
            circuit.rz(par[ind + 19], 8)
            circuit.rz(par[ind + 20], 9)
            circuit.rz(par[ind + 21], 10)

            for (i,j) in enumerate(itertools.combinations(range(9),2)):
                circuit.rzz(par[ind + 22+i], j[0], j[1])

        ind=11+(36+22)*L
        circuit.rx(par[ind + 0], 0)
        circuit.rx(par[ind + 1], 1)
        circuit.rx(par[ind + 2], 2)
        circuit.rx(par[ind + 3], 3)
        circuit.rx(par[ind + 4], 4)
        circuit.rx(par[ind + 5], 5)
        circuit.rx(par[ind + 6], 6)
        circuit.rx(par[ind + 7], 7)
        circuit.rx(par[ind + 8], 8)
        circuit.rx(par[ind + 9], 9)
        circuit.rx(par[ind + 10], 10)

        circuit.rz(par[ind + 11], 0)
        circuit.rz(par[ind + 12], 1)
        circuit.rz(par[ind + 13], 2)
        circuit.rz(par[ind + 14], 3)
        circuit.rz(par[ind + 15], 4)
        circuit.rz(par[ind + 16], 5)
        circuit.rz(par[ind + 17], 6)
        circuit.rz(par[ind + 18], 7)
        circuit.rz(par[ind + 19], 8)
        circuit.rz(par[ind + 20], 9)
        circuit.rz(par[ind + 21], 10)



        unitary=Operator(circuit).data[:,:2**self.n]
        for i in range(2**(11-self.n)):
            ind=i
            ele=unitary[ind*2**self.n:(ind+1)*2**self.n]
            out.append(ele)
        self.rec_kraus=out
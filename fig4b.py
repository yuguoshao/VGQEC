from vgqec.hybridscheme import VGQEC_five_hybrid
from vgqec.original_code import NoProtection,PerfectCode
from vgqec.environment import ThermalRelaxationNoise,ThermalRelaxationNoiseQ0
from vgqec import Optimizer
import numpy as np
import matplotlib.pyplot as plt
from fig4b_data import data

RUN=False
mission_list=[0,2,4,6,8,10]
OPT=[1,0.9956016516481846,0.9892200438066916,0.9811999068060324,0.971770282439932,0.9611676486415699]

if __name__ == '__main__':
    num_seeds=50
    code0=VGQEC_five_hybrid()
    code1=PerfectCode()
    code2=NoProtection()
    APD_Five=[1]
    APD_OFive=[1]
    Q0=[1]
    random_seeds = np.random.randint(0, 2 ** 32 - 1, size=num_seeds)
    for t in mission_list[1:]:
        env = ThermalRelaxationNoise(t)
        opt = Optimizer(code0, env)
        if RUN:
            temp=[opt.maximize_state_fidelity(seed=seed) for seed in random_seeds]
            max_first_item_tuple = max(temp, key=lambda x: x[0])
            APD_Five.append(opt.call_channel_fidelity(max_first_item_tuple[1]))
        else:
            APD_Five.append(opt.call_channel_fidelity(data[t//2-1][2]))
        APD_OFive.append(env.channel_fidelity(code1))

        env1 = ThermalRelaxationNoiseQ0(t)
        Q0.append(env1.channel_fidelity(code2))

    plt.plot(mission_list, APD_Five, label='Five-qubit VGQEC code', color='blue', marker='o')
    plt.plot(mission_list, APD_OFive, label='[[5,1,3]] code', linestyle='--', color='blue', marker='.')
    plt.plot(mission_list, Q0, color='red', linewidth=1.0, linestyle='--', label=r'$Q_0$', marker='.')
    plt.plot(mission_list, OPT, color='green', linewidth=1.0, linestyle='--', label='Numerically-optimized', marker='.')
    plt.grid(linestyle=":", color="black")
    plt.legend(loc='lower left')
    plt.xlabel(r'$t$')
    plt.ylabel('Channl Fidelity')
    plt.title('Decoherence Process')
    plt.show()
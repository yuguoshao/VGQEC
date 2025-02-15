from vgqec.light_vgqec import VGQEC_five_light
from vgqec.original_code import RepetitionCodeFive,NoProtection,PerfectCode
from vgqec.environment import CorrelatedNoise,CorrelatedNoiseSingle
from vgqec import Optimizer
import numpy as np
import matplotlib.pyplot as plt


qvector=[0.99839682,0.99755632,0.99664744,0.99440136,0.99391573,0.99329953,0.99302219,0.99248475,0.99210023,0.9914716,0.99094446]

if __name__ == '__main__':
    num_seeds=20
    code0=VGQEC_five_light()
    code1=PerfectCode()
    code2=RepetitionCodeFive()
    code3=NoProtection()
    eta=np.arange(0,1.1,0.1)
    SDP_opt=[]
    Rep_opt=[]
    Five_opt=[]
    no_protection=[]
    random_seeds = np.random.randint(0, 2 ** 32 - 1, size=num_seeds)
    for ele in eta:
        env = CorrelatedNoise(ele)
        opt1 = Optimizer(code0, env)
        temp=[opt1.maximize_optimal_fidelity(seed=seed)[0] for seed in random_seeds]
        SDP_opt.append(max(temp))
        Five_opt.append(env.optimal_fidelity(code1))
        Rep_opt.append(env.optimal_fidelity(code2))
        ad_env = CorrelatedNoiseSingle(ele)
        no_protection.append(ad_env.optimal_fidelity(code3))

    plt.figure(figsize=(4, 4))
    plt.plot(np.arange(0, 1.1, 0.1), SDP_opt, label='VGQEC code', color='green', marker='o')
    plt.plot(np.arange(0, 1.1, 0.1), Rep_opt, label='Five-qubit repetition code', linestyle='--', marker='.',
             color='blue')
    plt.plot(np.arange(0, 1.1, 0.1), Five_opt, label='[[5,1,3]] code', linestyle='--', marker='.', color='orange')
    plt.plot(np.arange(0, 1.1, 0.1), qvector, label='QVector', linestyle='--', marker='.', color='purple')
    plt.plot(np.arange(0, 1.1, 0.1), no_protection, color='red', linewidth=1.0, linestyle='--', label='No Protection',
             zorder=1)
    plt.grid(linestyle=":", color="black")
    plt.legend(loc='upper right')
    plt.xlabel(r'$\eta$')
    plt.ylim((0.989, 1.001))
    plt.show()
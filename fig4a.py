from vgqec.hybridscheme import VGQEC_three_hybrid,VGQEC_five_hybrid
from vgqec.original_code import RepetitionCode,NoProtection,PerfectCode
from vgqec.environment import AmplitudeDamping
from vgqec import Optimizer
import numpy as np
import matplotlib.pyplot as plt


ad_SDP_opt=[1.0000000000132916, 0.9856315736465662, 0.9675587732360057, 0.9458457045460967, 0.9205793713551361, 0.8918723343903847, 0.8598650992254708, 0.8247283826111335, 0.7866654576699541, 0.7583099241519493]

if __name__ == '__main__':
    num_seeds=50
    code0=VGQEC_five_hybrid()
    code1=VGQEC_three_hybrid()
    code2=PerfectCode()
    code3=RepetitionCode()
    code4=NoProtection()
    lam=np.arange(0,0.5,0.05)
    AD_MOD_Five=[]
    AD_MOD_Three=[]
    npAD_Five=[]
    npAD_Three=[]
    no_protection=[]
    random_seeds = np.random.randint(0, 2 ** 32 - 1, size=num_seeds)
    for ele in lam:
        env5 = AmplitudeDamping(5,ele)
        opt1 = Optimizer(code0, env5)
        temp=[opt1.maximize_state_fidelity(seed=seed) for seed in random_seeds]
        max_first_item_tuple = max(temp, key=lambda x: x[0])
        AD_MOD_Five.append(opt1.call_channel_fidelity(max_first_item_tuple[1]))
        npAD_Five.append(env5.channel_fidelity(code2))

        env3 = AmplitudeDamping(3, ele)
        opt2 = Optimizer(code1, env3)
        temp = [opt2.maximize_state_fidelity(seed=seed) for seed in random_seeds]
        max_first_item_tuple = max(temp, key=lambda x: x[0])
        AD_MOD_Three.append(opt2.call_channel_fidelity(max_first_item_tuple[1]))
        npAD_Three.append(env3.channel_fidelity(code3))

        ad_env = AmplitudeDamping(1,ele)
        no_protection.append(ad_env.channel_fidelity(code4))


    fig, main_ax = plt.subplots()
    main_ax.plot(np.arange(0, 0.5, 0.05), AD_MOD_Five, label='Five-qubit VGQEC code', color='blue', marker='o',
                 zorder=6)

    main_ax.plot(np.arange(0, 0.5, 0.05), AD_MOD_Three, label='Three-qubit VGQEC code', color='orange', marker='o',
                 zorder=5)

    main_ax.plot(np.arange(0, 0.5, 0.05), npAD_Five, label='[[5,1,3]] code', linestyle='--', color='blue',
                 marker='.', zorder=4)
    main_ax.plot(np.arange(0, 0.5, 0.05), npAD_Three, label='Three-qubit repetition code', linestyle='--',
                 color='orange', marker='.', zorder=3)
    main_ax.plot(np.arange(0, 0.5, 0.05), ad_SDP_opt, label='Fig. 2 (optimal channel fidelity)', linestyle='--',
                 color='green', zorder=5)
    main_ax.plot(np.arange(0, 0.5, 0.05), no_protection, color='red', linewidth=1.0, linestyle='--',
                 label='No Protection', zorder=1)
    main_ax.grid(linestyle=":", color="black")

    main_ax.legend(loc='lower left')
    main_ax.set_ylim((0.74, 1.01))
    main_ax.set_xlabel(r'$\lambda$')
    main_ax.set_ylabel('Channel Fidelity')
    main_ax.set_title('Amplitude Damping Noise')
    plt.show()
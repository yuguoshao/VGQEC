from vgqec.light_vgqec import VGQEC_three_light
from vgqec.original_code import RepetitionCode,NoProtection
from vgqec.environment import AmplitudeDamping
from vgqec import Optimizer
import numpy as np
import matplotlib.pyplot as plt

ad3_iter=[0.9266574869648865, 0.9675738136877006, 0.9675738140790766]
qvector_iter=[0.9363280568841893, 0.9509250195994061, 0.9575406503072939, 0.9598772575753256, 0.9601272993305181, 0.9605272562595331, 0.9607024865029505, 0.9608627672039336, 0.9609332403632628, 0.9609461807408592, 0.9609507970477544, 0.9609512412859407]
qvector=[1.0000000000000238, 0.9817881210935375, 0.9609512412859407, 0.9376169045922818, 0.9119297642669759, 0.8840582268668787, 0.8542033675860676, 0.8226126334146957, 0.7896308580752758, 0.7562026143376951]

if __name__ == '__main__':
    num_seeds=20
    code0=VGQEC_three_light()
    code1=RepetitionCode()
    code2=NoProtection()
    lam=np.arange(0,0.5,0.05)
    ad_SDP_opt=[]
    ad_three=[]
    ad_single=[]
    random_pert = np.random.randn(num_seeds) * 1e-3
    for ele in lam:
        ad_env = AmplitudeDamping(3, ele)
        opt1 = Optimizer(code0, ad_env)
        temp=[opt1.maximize_optimal_fidelity(init_params=np.array(alpha))[0] for alpha in random_pert]
        ad_SDP_opt.append(max(temp))
        ad_three.append(ad_env.optimal_fidelity(code1))
        ad_env = AmplitudeDamping(1, ele)
        ad_single.append(ad_env.optimal_fidelity(code2))


    fig, main_ax = plt.subplots(figsize=(4, 4))
    main_ax.plot(lam, ad_SDP_opt, label='VGQEC code', color='green', marker='o')
    main_ax.plot(lam, ad_three, label='Three-qubit repetition code', linestyle='--', marker='.',
                 color='blue')
    main_ax.plot(np.arange(0,0.5,0.05), qvector, label='QVector', linestyle='--', marker='.', color='purple')
    main_ax.plot(lam, ad_single, color='red', linewidth=1.0, linestyle='--', label='No Protection',
                 zorder=1)
    right_inset_ax = fig.add_axes([.65, .55, .25, .3])
    right_inset_ax.plot([0, 1, 2], ad3_iter, color='green', linewidth=1.0, linestyle='-', label='VGQEC')
    right_inset_ax.plot(range(len(qvector_iter)), qvector_iter, color='purple', linewidth=1.0, linestyle='--',
                        label='QVector')
    right_inset_ax.set(xticks=[0, 5, 10])
    right_inset_ax.grid(linestyle=":", color="black")
    right_inset_ax.set_xlabel('Steps')

    main_ax.grid(linestyle=":", color="black")
    main_ax.legend(loc='lower left')
    main_ax.set_ylim((0.74, 1.01))
    main_ax.set_xlabel(r'$\lambda$')
    main_ax.set_ylabel('Optimal Channel Fidelity')
    plt.show()
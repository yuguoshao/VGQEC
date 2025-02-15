import vgqec
import numpy as np

code0=vgqec.CodeBase()
code0.k,code0.n=1,1
code0.encode_mat=np.identity(2)
code0.rec_kraus=[np.identity(2)]

code1=vgqec.light_vgqec.VGQEC_three_light()
code1.set_parameters([1.57087988])
print(code1.encode_mat)

code2=vgqec.light_vgqec.VGQEC_five_light()
code2.set_parameters([0,0,0,0,0])
code2.set_parameters([-0.5*np.pi,-0.5*np.pi,-0.5*np.pi,-0.5*np.pi,-0.5*np.pi])
print((code2.encode_mat*8/(-1-1j)).real)
print(code2.basecode.encode_mat*np.sqrt(32)-(code2.encode_mat*8/(-1-1j)).real)
print(code2.basecode.check_kraus())

code3=vgqec.hybridscheme.VGQEC_three_hybrid()
ad_env=vgqec.environment.AmplitudeDamping(1,0.05)
ad_env.channel_fidelity(code0)
import vgqec
import numpy as np
code1=vgqec.light_vgqec.VGQEC_three_light()
code1.set_parameters([1.57087988])
print(code1.encode_mat)

code2=vgqec.light_vgqec.VGQEC_five_light()
code2.set_parameters([0,0,0,0,0])
code2.set_parameters([-0.5*np.pi,-0.5*np.pi,-0.5*np.pi,-0.5*np.pi,-0.5*np.pi])
print((code2.encode_mat*8/(-1-1j)).real)
print(code2.basecode.encode_mat*np.sqrt(32)-(code2.encode_mat*8/(-1-1j)).real)
print(code2.check_kraus())

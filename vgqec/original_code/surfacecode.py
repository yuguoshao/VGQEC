import numpy as np
from . import CodeBase


N_PHYS = 9  # data qubits
N_TOTAL = 10  # data(0..8) + reference(9)

def pstr_from_ops(ops, n=N_PHYS):
    s = ["I"] * n
    for q, op in ops.items():
        s[q] = op
    return "".join(s)

# 4 个 X-check + 4 个 Z-check（示例：全部 weight-4）
STABILIZERS_X = [
    pstr_from_ops({0:"X", 1:"X", 3:"X", 4:"X"}),
    pstr_from_ops({1:"X", 2:"X", 4:"X", 5:"X"}),
    pstr_from_ops({3:"X", 4:"X", 6:"X", 7:"X"}),
    pstr_from_ops({4:"X", 5:"X", 7:"X", 8:"X"}),
]
STABILIZERS_Z = [
    pstr_from_ops({0:"Z", 1:"Z", 3:"Z", 4:"Z"}),
    pstr_from_ops({1:"Z", 2:"Z", 4:"Z", 5:"Z"}),
    pstr_from_ops({3:"Z", 4:"Z", 6:"Z", 7:"Z"}),
    pstr_from_ops({4:"Z", 5:"Z", 7:"Z", 8:"Z"}),
]
STABILIZERS = STABILIZERS_X + STABILIZERS_Z  # 8 个

LOGICAL_Z = pstr_from_ops({0:"Z", 1:"Z", 2:"Z"})   # 示例 logical Z
LOGICAL_X = pstr_from_ops({0:"X", 3:"X", 6:"X"})   # 示例 logical X

# ============================================================
# 1) Pauli 字符串 <-> (xmask,zmask,ny) + 共轭作用的快速实现
# ============================================================

def pauli_to_masks(pstr: str):
    xmask = 0
    zmask = 0
    ny = 0
    for i, ch in enumerate(pstr):
        bit = 1 << i
        if ch == "X":
            xmask |= bit
        elif ch == "Z":
            zmask |= bit
        elif ch == "Y":
            xmask |= bit
            zmask |= bit
            ny += 1
        elif ch == "I":
            pass
        else:
            raise ValueError(f"Bad Pauli char: {ch}")
    return xmask, zmask, ny

def symplectic_anticommutes(p_mask, q_mask):
    (xp, zp, _), (xq, zq, _) = p_mask, q_mask
    return (((xp & zq).bit_count() + (zp & xq).bit_count()) & 1) == 1

def popcount_u16(arr_u16: np.ndarray) -> np.ndarray:
    b = arr_u16.view(np.uint8).reshape(-1, 2)
    bits = np.unpackbits(b, axis=1)
    return bits.sum(axis=1).astype(np.int32)

def precompute_perm_phase(xmask, zmask, ny, n=N_PHYS):
    dim = 1 << n
    idx = np.arange(dim, dtype=np.uint16)
    src = idx ^ np.uint16(xmask)
    pc = popcount_u16((src & np.uint16(zmask)).astype(np.uint16))
    phase = (1j ** ny) * ((-1) ** pc)
    return src.astype(np.int32), phase.astype(np.complex128)

def apply_pauli_to_statevec(v, src, phase):
    return phase * v[src]

def conj_block_by_pauli(A, src, phase):
    tmp = phase[:, None] * A[src, :]
    out = tmp[:, src] * phase.conj()[None, :]
    return out

def conj_full_by_pauli_on_phys(rho10, src, phase):
    # rho10 acts on (phys 0..8, ref 9).  We reshape as (ref,phys,ref,phys)
    # but since ref is MSB (qubit 9), indices split into [phys | ref].
    rho4 = rho10.reshape(2, 1<<N_PHYS, 2, 1<<N_PHYS)
    tmp = rho4[:, src, :, :] * phase[None, :, None, None]
    out = tmp[:, :, :, src] * phase.conj()[None, None, None, :]
    return out.reshape(1<<N_TOTAL, 1<<N_TOTAL)

# ============================================================
# 2) 构造 |0_L>, |1_L> 与 codespace isometry U (512x2)
#    这里用 “从 |0...0> 投影到 +1 eigenspace of X-stabilizers”
# ============================================================

def make_logical_basis():
    dim = 1 << N_PHYS
    v = np.zeros(dim, dtype=np.complex128)
    v[0] = 1.0

    # v <- Π_i (I+Sx_i) v
    for sx in STABILIZERS_X:
        xmask, zmask, ny = pauli_to_masks(sx)
        src, phase = precompute_perm_phase(xmask, zmask, ny)
        v = v + apply_pauli_to_statevec(v, src, phase)

    v = v / np.linalg.norm(v)

    # |1_L> = X_L |0_L>
    xmask, zmask, ny = pauli_to_masks(LOGICAL_X)
    src, phase = precompute_perm_phase(xmask, zmask, ny)
    v1 = apply_pauli_to_statevec(v, src, phase)

    # 数值正交化
    v1 = v1 - np.vdot(v, v1) * v
    v1 = v1 / np.linalg.norm(v1)

    return v, v1



# ============================================================
# 3) pure-error 生成元：对每个 stabilizer 找一个 Pauli E_i
#    满足：只与该 stabilizer 反对易，其它对易（weight<=2 搜索）
# ============================================================

STAB_MASKS = [pauli_to_masks(s) for s in STABILIZERS]

def find_pure_error_for(i, max_weight=2):
    target = [0]*len(STAB_MASKS)
    target[i] = 1

    # weight-1
    for q in range(N_PHYS):
        for op in ["X","Y","Z"]:
            p = pstr_from_ops({q: op})
            pm = pauli_to_masks(p)
            comm = [int(symplectic_anticommutes(pm, sm)) for sm in STAB_MASKS]
            if comm == target:
                return p

    # weight-2
    if max_weight >= 2:
        for q1 in range(N_PHYS):
            for q2 in range(q1+1, N_PHYS):
                for op1 in ["X","Y","Z"]:
                    for op2 in ["X","Y","Z"]:
                        p = pstr_from_ops({q1: op1, q2: op2})
                        pm = pauli_to_masks(p)
                        comm = [int(symplectic_anticommutes(pm, sm)) for sm in STAB_MASKS]
                        if comm == target:
                            return p

    raise RuntimeError(f"Cannot find pure error for stabilizer {i} within weight<= {max_weight}")

PURE_ERRORS = [find_pure_error_for(i, max_weight=2) for i in range(len(STABILIZERS))]
PURE_PRECOMP = []
for pe in PURE_ERRORS:
    xmask, zmask, ny = pauli_to_masks(pe)
    src, phase = precompute_perm_phase(xmask, zmask, ny)
    PURE_PRECOMP.append((src, phase))



class SurfaceCode9(CodeBase):
    def __init__(self):
        super().__init__()
        self.n=9
        self.k=1
        self.encode_mat=self.encode_mat_fun()
        self.init_gen()
        self.gen_rec_kraus()


    def gen_rec_kraus(self):
        num=3
        n = 2 ** num
        krauses = []
        for i in range(int(n / 2)):
            kraus = np.zeros([n, n])
            if bin(i).count('1') < num / 2:
                kraus[0][i] = 1
                kraus[n - 1][n - i - 1] = 1
            else:
                kraus[n - 1][i] = 1
                kraus[0][n - i - 1] = 1
            krauses.append(kraus)
        self.rec_kraus= krauses

    def encode_mat_fun(self):
        res = np.array([make_logical_basis])
        return res

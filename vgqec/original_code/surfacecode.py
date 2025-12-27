import numpy as np
from . import CodeBase
import pymatching

N_PHYS = 9  # data qubits
N_TOTAL = 10  # data(0..8) + reference(9)

def pstr_from_ops(ops, n=N_PHYS):
    s = ["I"] * n
    for q, op in ops.items():
        s[q] = op
    return "".join(s)

# 4 个 X-check + 4 个 Z-check（示例：全部 weight-4）
STABILIZERS_X = [
    pstr_from_ops({0:"X", 1:"X"}),
    pstr_from_ops({1:"X", 2:"X", 4:"X", 5:"X"}),
    pstr_from_ops({3:"X", 4:"X", 6:"X", 7:"X"}),
    pstr_from_ops({7:"X", 8:"X"}),
]
STABILIZERS_Z = [
    pstr_from_ops({0:"Z", 1:"Z", 3:"Z", 4:"Z"}),
    pstr_from_ops({2:"Z", 5:"Z"}),
    pstr_from_ops({3:"Z", 6:"Z"}),
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



def build_check_matrices(Sx_pstrs, Sz_pstrs, n):
    mX = len(Sx_pstrs)
    mZ = len(Sz_pstrs)
    H_Z = np.zeros((mX, n), dtype=np.uint8)  # for decoding Z errors
    H_X = np.zeros((mZ, n), dtype=np.uint8)  # for decoding X errors

    # Z on qubit q anticommutes with X/Y on that qubit
    for i, sx in enumerate(Sx_pstrs):
        for q in range(n):
            ch = sx[q]
            if ch in ("X", "Y"):
                H_Z[i, q] ^= 1

    # X on qubit q anticommutes with Z/Y on that qubit
    for i, sz in enumerate(Sz_pstrs):
        for q in range(n):
            ch = sz[q]
            if ch in ("Z", "Y"):
                H_X[i, q] ^= 1

    return H_Z, H_X

def build_kraus_with_pymatching_mw(Sx_pstrs, Sz_pstrs, U):
    n = len(Sx_pstrs[0])
    stabs = list(Sx_pstrs) + list(Sz_pstrs)
    m = len(stabs)
    #U     # 2x512

    # ---- PyMatching decoders from check matrices ----
    H_Z, H_X = build_check_matrices(Sx_pstrs, Sz_pstrs, n)

    # Uniform weights => MW by number of qubits (ties arbitrary but still MW).
    # Matching can be built from a check matrix directly. :contentReference[oaicite:4]{index=4}
    decZ = pymatching.Matching(H_Z)  # decode Z errors using X syndromes
    decX = pymatching.Matching(H_X)  # decode X errors using Z syndromes

    Ks = []
    for syn_int in range(1 << m):
        # split syndrome bits
        synX = np.array([(syn_int >> i) & 1 for i in range(len(Sx_pstrs))], dtype=np.uint8)
        synZ = np.array([(syn_int >> (len(Sx_pstrs) + i)) & 1 for i in range(len(Sz_pstrs))], dtype=np.uint8)

        # MWPM corrections (bitvector over data qubits)
        z_corr = decZ.decode(synX)  # length n, 1 means apply Z on that qubit  :contentReference[oaicite:5]{index=5}
        x_corr = decX.decode(synZ)  # length n, 1 means apply X on that qubit

        # build correction Pauli masks
        xmask = 0
        zmask = 0
        for q in range(n):
            if int(x_corr[q]):
                xmask |= (1 << q)
            if int(z_corr[q]):
                zmask |= (1 << q)
        ny = (xmask & zmask).bit_count()
        C_pre = precompute_perm_phase(xmask, zmask, ny, n=n)
        src, phase = C_pre
        # K_s = U^† C_s Π_s:
        # build it by acting on |0_L>,|1_L> (avoid 512x512 projector)
        w0 = apply_pauli_to_statevec(U[0], src, phase)
        w1 = apply_pauli_to_statevec(U[1], src, phase)
        K = np.stack([w0.conj(), w1.conj()], axis=0)  # (2,512)
        Ks.append(K)

    # sanity: completeness Σ K†K = I
    M = np.zeros((1 << n, 1 << n), dtype=np.complex128)
    for K in Ks:
        M += K.conj().T @ K
    #print("[check] max |Σ K†K - I| =", np.max(np.abs(M - np.eye(1 << n))))
    assert np.allclose(M, np.eye(1 << n))

    return Ks


class SurfaceCode9(CodeBase):
    def __init__(self):
        super().__init__()
        self.n=9
        self.k=1
        self.encode_mat=self.encode_mat_fun()
        self.init_gen()
        self.gen_rec_kraus()


    def gen_rec_kraus(self):
        krauses = build_kraus_with_pymatching_mw(STABILIZERS_X, STABILIZERS_Z, self.encode_mat)
        self.rec_kraus= krauses

    def encode_mat_fun(self):
        res = np.array(make_logical_basis())
        return res

if __name__ == '__main__':
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # 加入父目录到路径中
    import sys

    sys.path.append(os.path.dirname(dir_path))
    from codebase import CodeBase
    code = SurfaceCode9()
    print(code.encode_mat)
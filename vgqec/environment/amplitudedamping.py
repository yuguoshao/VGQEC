from .envbase import EnvBase
from qiskit_aer import noise as noise_aer
import qiskit

import numpy as np
from itertools import product
from functools import reduce

def amp_damp_single_kraus(lam: float):
    lam = float(lam)
    K0 = np.array([[1.0, 0.0],
                   [0.0, np.sqrt(1.0 - lam)]], dtype=complex)
    K1 = np.array([[0.0, np.sqrt(lam)],
                   [0.0, 0.0]], dtype=complex)
    return K0, K1


class AmplitudeDamping(EnvBase):
    def __init__(self,n,lam=0.1):
        super().__init__()
        self.n=n
        self.lam=lam
        self._local_kraus=None
        self.set_noise()

    def set_noise(self):
        K0, K1 = amp_damp_single_kraus(self.lam)
        ops = (K0, K1)

        def kron_qiskit_order(mats_for_q0_to_qn1):
            # Qiskit order: leftmost factor is qubit n-1, rightmost is qubit 0
            return reduce(np.kron, mats_for_q0_to_qn1[::-1])

        iterable = product([0, 1], repeat=self.n)

        local_kraus = []
        ident = np.eye(2, dtype=complex)
        for q in range(self.n):
            mats0 = [ident] * self.n
            mats1 = [ident] * self.n
            mats0[q] = K0
            mats1[q] = K1
            L0 = kron_qiskit_order(mats0)
            L1 = kron_qiskit_order(mats1)
            local_kraus.append((L0, L0.conj().T, L1, L1.conj().T))
        self._local_kraus = local_kraus

        out = []
        for bits in iterable:
            mats = [ops[b] for b in bits]
            out.append(kron_qiskit_order(mats))
        self.noise_kraus=out

    def evo_mat(self, density_matrix):
        # Independent amplitude damping on n qubits can be applied qubit-by-qubit.
        # This is equivalent to summing over 2^n global Kraus operators but much faster.
        res = np.asarray(density_matrix, dtype=np.complex128)
        for L0, L0_dag, L1, L1_dag in self._local_kraus:
            res = L0 @ res @ L0_dag + L1 @ res @ L1_dag
        return res

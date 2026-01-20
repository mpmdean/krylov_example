import numpy as np
from scipy.sparse import coo_matrix
from math import comb, prod

from dataclasses import dataclass, field
from typing import List, Tuple
from itertools import accumulate


@dataclass(slots=True)
class FockBinByN:
    """Basis class to handle the product space of fixed-particle-number
    Fock subspaces.

    Parameters
    ----------
    shapes
        List of ``(norb, noccu)`` pairs, one per subspace.

    Attributes
    ----------
    norbs
        Number of orbitals in each subspace.
    noccus
        Number of particles in each subspace.
    sizes
        Dimension of each subspace, ``comb(norb[i], noccu[i])``.
    num_subspaces
        Number of subspaces
    num_orbitals
        Number of orbitals
    dim
        Number of states in basis
    """
    shapes: List[Tuple[int, int]]
    norbs: List[int] = field(init=False)
    noccus: List[int] = field(init=False)
    sizes: List[int] = field(init=False)

    num_subspaces: int = field(init=False)
    num_orbitals: int = field(init=False)
    dim: int = field(init=False)

    offsets: Tuple[int, ...] = field(init=False)
    strides: Tuple[int, ...] = field(init=False)
    min_decode: int = field(init=False)
    max_decode: int = field(init=False)

    def __post_init__(self) -> None:
        self.norbs = [int(norb) for (norb, _) in self.shapes]
        self.noccus = [int(nocc) for (_, nocc) in self.shapes]
        self.sizes = [comb(norb, nocc) for (norb, nocc) in self.shapes]
        self.num_subspaces = len(self.norbs)
        self.num_orbitals = sum(self.norbs)
        self.dim = prod(self.sizes)

        running = self.num_orbitals
        offsets = []
        for N in self.norbs:
            running -= N
            offsets.append(running)
        self.offsets = tuple(offsets)

        extrema = min_max_decode(self.shapes)
        self.min_decode = extrema[0]
        self.max_decode = extrema[1]

        stride = 1
        strides = [0] * self.num_subspaces
        for n in reversed(range(self.num_subspaces)):
            strides[n] = stride
            stride *= self.sizes[n]
        self.strides = tuple(strides)


    def encode(self, b: int) -> int:
        """Encode occupations into a basis index.
    
        Parameters
        ----------
        b: int
            0/1 occupation vector encoded as an integer
    
        Returns
        -------
        index
            Basis index in ``[0, space.dim - 1]``.
        """
        index = 0
        if b < self.min_decode or b > self.max_decode:
            return -1

        for n in range(self.num_subspaces):
            start = self.offsets[n]
            N = self.norbs[n]
            M = self.noccus[n]
            mask = (1 << N) - 1

            sub_bits = (b >> start) & mask
            if sub_bits.bit_count() != M:
                return -1

            sub_rank = hash_encoder(sub_bits, N, M)
            index += sub_rank * self.strides[n]

        return index

    def decode(self, index: int) -> int:
        """Decode a basis index into occupations.
    
        Parameters
        ----------
        index
            basis index in ``[0, space.dim - 1]``.
    
        Returns
        -------
        b: int
            0/1 occupation vector encoded as an integer

        """
        b = 0

        for n in reversed(range(self.num_subspaces)):
            d = self.sizes[n]
            sub_rank = index % d
            index //= d

            sub_bits = hash_decoder(sub_rank, self.norbs[n], self.noccus[n])
            b |= (sub_bits << self.offsets[n])

        return b


def hash_decoder(r: int, N: int, M: int) -> int:
    """Unrank a bitstring encoded as an integer.

    Parameters
    ----------
    r: int
        Rank in ``[0, comb(N, M) - 1]``.
    N: int
        Bitstring length (number of orbitals).
    M: int
        Number of ones.

    Returns
    -------
    b: int
        0/1 occupation vector encoded as an integer

    """
    b = 0
    j = M
    i = N - 1

    c = comb(i, j) # MPMD store me?
    while i >= 0 and j > 0:
        if c <= r:
            b |= (1 << i)
            r -= c
            old_i, old_j = i, j
            i -= 1
            j -= 1
            if j == 0 or i < 0:
                break
            c = (c * old_j) // old_i
        else:
            if i == 0:
                break
            c = (c * (i - j)) // i
            i -= 1

    return b


def hash_encoder(b: int, N: int, M: int) -> int:
    """Rank a fixed-weight bitstring encoded as an integer.

    Parameters
    ----------
    b:int
        0/1 occupation vector encoded as an integer
    N: int
        Bitstring length (number of orbitals).
    M: int
        Number of ones.

    Returns
    -------
    int
        Rank of the state.
    """
    b = int(b) & ((1 << N) - 1)

    r = 0
    k = 1
    while b:
        lsb = b & -b
        pos = lsb.bit_length() - 1
        r += comb(pos, k)
        k += 1
        b ^= lsb
    return r


def sign_count(b: int, orb: int, norb: int) -> int:
    """(-1)**(number of 1) up to position orb
    in b."""
    bitpos = norb - 1 - orb
    prefix = b >> (bitpos + 1)
    return 1 if (prefix.bit_count() % 2 == 0) else -1


def bit_at_orb(b: int, orb: int, norb: int) -> int:
    """
    Return the bit value at orbital `orb` in b
    """
    bitpos = norb - 1 - orb
    return (b >> bitpos) & 1


def set_zero_at_orb(b: int, orb: int, norb: int) -> int:
    """
    Return a new integer where b[orb] == 0,
    """
    bitpos = norb - 1 - orb
    return b & ~(1 << bitpos)


def set_one_at_orb(b: int, orb: int, norb: int) -> int:
    """
    Return a new integer where b[orb] == 1,
    """
    bitpos = norb - 1 - orb
    return b | (1 << bitpos)


def two_fermion_B(emat, lb, rb=None, tol=1E-10):
    """
    Build the csr sparse matrix form of a two-fermionic operator
    in the given Fock basis,

    .. math::

        <F_{l}|\\sum_{ij}E_{ij}\\hat{f}_{i}^{\\dagger}\\hat{f}_{j}|F_{r}>

    Parameters
    ----------
    emat: 2d complex array
        The impurity matrix.
    lb: FockBinByN
        Left fock basis :math:`<F_{l}|`.
    rb: FockBinByN
        Right fock basis :math:`|F_{r}>`.
        rb = lb if rb is None
    tol: float (default: 1E-10)
        Only consider the elements of emat that are larger than tol.

    Returns
    -------
    indptr, indices, data, nl, nr: various numpy arrays
        The matrix form of the two-fermionic operator expressed as csr
    """
    if rb is None:
        rb = lb

    a1, a2 = np.nonzero(abs(emat) > tol)
    nonzero = np.stack((a1, a2), axis=-1)

    rows = []
    cols = []
    data = []
    norb = rb.num_orbitals
    
    for iorb, jorb in nonzero:
        for icfg in range(rb.dim):
            b = rb.decode(icfg)
            if bit_at_orb(b, jorb, norb) == 0:
                continue
            else:
                s1 = sign_count(b, jorb, norb)
                b = set_zero_at_orb(b, jorb, norb)
            if bit_at_orb(b, iorb, norb) == 1:
                continue
            else:
                s2 = sign_count(b, iorb, norb)
                b = set_one_at_orb(b, iorb, norb)
            jcfg = lb.encode(b)
            if jcfg != -1:
                 rows.append(jcfg)
                 cols.append(icfg)
                 data.append(emat[iorb, jorb] * s1 * s2)

    # We want to interface between the functions using csr. Eventually, we need to generate 
    # these directly, for now, let scipy convert.
    A = coo_matrix((data, (rows, cols)), shape=(lb.dim, rb.dim), dtype=np.complex128).tocsr()
    return A.indptr, A.indices, A.data, lb.dim, rb.dim


def four_fermion_B(umat, lb, rb=None, tol=1E-10):
    """
    Build the csr sparse  matrix form of a four-fermionic operator
    in the given Fock basis,

    .. math::

        <F_l|\\sum_{ij}U_{ijkl}\\hat{f}_{i}^{\\dagger}\\hat{f}_{j}^{\\dagger}
        \\hat{f}_{k}\\hat{f}_{l}|F_r>

    Parameters
    ----------
    umat: 4d complex array
        The 4 index Coulomb interaction tensor.
    lb: FockBinByN
        Left fock basis :math:`<F_{l}|`.
    rb: FockBinByN
        Right fock basis :math:`|F_{r}>`.
        rb = lb if rb is None
    tol: float (default: 1E-10)
        Only consider the elements of umat that are larger than tol.

    Returns
    -------
    indptr, indices, data, nl, nr: various numpy arrays
        The matrix form of the two-fermionic operator expressed as csr
    """
    if rb is None:
        rb = lb

    a1, a2, a3, a4 = np.nonzero(abs(umat) > tol)
    nonzero = np.stack((a1, a2, a3, a4), axis=-1)

    rows = []
    cols = []
    data = []
    norb = rb.num_orbitals

    for lorb, korb, jorb, iorb in nonzero:
        if iorb == jorb or korb == lorb:
            continue
        for icfg in range(rb.dim):
            b = rb.decode(icfg)
            if bit_at_orb(b, iorb, norb) == 0:
                continue
            else:
                s1 = sign_count(b, iorb, norb)
                b = set_zero_at_orb(b, iorb, norb)
            if bit_at_orb(b, jorb, norb) == 0:
                continue
            else:
                s2 = sign_count(b, jorb, norb)
                b = set_zero_at_orb(b, jorb, norb)
            if bit_at_orb(b, korb, norb) == 1:
                continue
            else:
                s3 = sign_count(b, korb, norb)
                b = set_one_at_orb(b, korb, norb)
            if bit_at_orb(b, lorb, norb) == 1:
                continue
            else:
                s4 = sign_count(b, lorb, norb)
                b = set_one_at_orb(b, lorb, norb)
            jcfg = lb.encode(b)
            if jcfg != -1:
                 rows.append(jcfg)
                 cols.append(icfg)
                 data.append(umat[lorb, korb, jorb, iorb] * s1 * s2 * s3 * s4)

    # We want to interface between the functions using csr. Eventually, we need to generate 
    # these directly, for now, let scipy convert.
    A = coo_matrix((data, (rows, cols)), shape=(lb.dim, rb.dim), dtype=np.complex128).tocsr()
    return A.indptr, A.indices, A.data, lb.dim, rb.dim


def min_max_decode(shapes):
    """Smallest and largest decoding values for a specific
    shape. Needed to block attempts to rank integers out 
    of range."""
    Ns = [N for N, _ in shapes]
    totalN = sum(Ns)

    b_min = 0
    b_max = 0
    running = totalN
    for (N, M) in shapes:
        running -= N
        sub_min = (1 << M) - 1
        sub_max = ((1 << M) - 1) << (N - M)
        b_min |= sub_min << running
        b_max |= sub_max << running

    return b_min, b_max
 

from petsc4py import PETSc

def get_H_emat(emat, lb, rb=None):
    indptr, indices, data, nl, nr = two_fermion_B(emat, lb, rb)
    return PETSc.Mat().createAIJ(size=(nl, nr), csr=(indptr, indices, data))


def get_H_umat(umat, lb, rb=None):
    indptr, indices, data, nl, nr = four_fermion_B(umat, lb, rb)
    return PETSc.Mat().createAIJ(size=(nl, nr), csr=(indptr, indices, data))


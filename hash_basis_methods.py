import numpy as np
from scipy.sparse import coo_matrix
from math import comb, prod

from dataclasses import dataclass, field
from typing import List, Tuple


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
    norb
        Number of orbitals in each subspace.
    noccu
        Number of particles in each subspace.
    sizes
        Dimension of each subspace, ``comb(norb[i], noccu[i])``.
    num_subspaces
        Number of subspaces
    num_orbitals
        MNmber of orbitals
    dim
        Number of states in basis
    """
    shapes: List[Tuple[int, int]]
    norb: List[int] = field(init=False)
    noccu: List[int] = field(init=False)
    sizes: List[int] = field(init=False)

    num_subspaces: int = field(init=False)
    num_orbitals: int = field(init=False)
    dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.norb = [int(norb) for (norb, _) in self.shapes]
        self.noccu = [int(nocc) for (_, nocc) in self.shapes]
        self.sizes = [comb(norb, nocc) for (norb, nocc) in self.shapes]
        self.num_subspaces = len(self.norb)
        self.num_orbitals = sum(self.norb)
        self.dim = prod(self.sizes)


    def encode(self, occupations: List[int]) -> int:
        """Encode occupations into a basis index.
    
        Parameters
        ----------
        occupations
            Concatenated 0/1 occupation vector across all subspaces.
    
        Returns
        -------
        index
            Basis index in ``[0, space.dim - 1]``.
        """
        start = 0
        sub_ranks: List[int] = [0] * self.num_subspaces
    
        for n in range(self.num_subspaces):
            norb = self.norb[n]
            sub_ranks[n] = hash_encoder([x for x in occupations[start : start + norb]])
            start += norb

        index = 0
        stride = 1
        for n in range(self.num_subspaces):
            index += sub_ranks[n] * stride
            stride *= self.sizes[n]

        # return -1 if occupations is not in basis
        if index == 0:
            if occupations != self.decode(0):
                return -1
            
        return index

    def decode(self, index: int) -> List[int]:
        """Decode a basis index into occupations.
    
        Parameters
        ----------
        index
            basis index in ``[0, space.dim - 1]``.
    
        Returns
        -------
        list of int
            Concatenated 0/1 occupation vector across all subspaces.

        """
        occupations: List[int] = []
        for n in range(self.num_subspaces):
            d = self.sizes[n]
            sub_rank = index % d
            index //= d
            occupations.extend(hash_decoder(sub_rank, self.norb[n], self.noccu[n]))
    
        return occupations


def hash_decoder(r: int, N: int, M: int) -> List[int]:
    """Unrank a fixed-weight bitstring.

    Parameters
    ----------
    r
        Rank in ``[0, comb(N, M) - 1]``.
    N
        Bitstring length (number of orbitals).
    M
        Number of ones.

    Returns
    -------
    list of int
        Length-``N`` list of 0/1 values with exactly ``M`` ones.

    """
    N = int(N)
    M = int(M)
    rank = int(r)

    total = comb(N, M)
    if rank < 0 or rank >= total:
        raise ValueError(f"r must be in [0, {total - 1}], got {r}.")

    if M == 0:
        return [0] * N
    if M == N:
        return [1] * N

    occ = [0] * N
    j = M
    i = N - 1

    c = comb(i, j)

    while i >= 0 and j > 0:
        if c <= rank:
            occ[i] = 1
            rank -= c
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

    return occ


def hash_encoder(occ: List[int]) -> int:
    """Rank a fixed-weight bitstring.

    Parameters
    ----------
    occ
        0/1 occupation vector.

    Returns
    -------
    int
        Rank of the state.

    Notes
    -----
    The rank ordering is consistent with :func:`hash_decoder`, i.e.
    ``hash_decoder(hash_encoder(occ), N, sum(occ)) == occ`` for valid inputs.
    """
    M = len(occ)
    r = 0
    j = 1  # next '1' will be the j-th one
    c = 0

    for i in range(M):
        if int(occ[i]) != 0:
            c_old = c
            r += c_old
            j += 1
            c = (c_old * (i + 1)) // j
        else:
            if j > i + 1:
                c = 0
            elif j == i + 1:
                c = 1
            else:
                c = (c * (i + 1)) // (i + 1 - j)

    return r


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

    tmp_basis = []*rb.num_orbitals
    for iorb, jorb in nonzero:
        for icfg in range(rb.dim):
            tmp_basis[:] = rb.decode(icfg)
            if tmp_basis[jorb] == 0:
                continue
            else:
                s1 = (-1)**np.count_nonzero(tmp_basis[0:jorb])
                tmp_basis[jorb] = 0
            if tmp_basis[iorb] == 1:
                continue
            else:
                s2 = (-1)**np.count_nonzero(tmp_basis[0:iorb])
                tmp_basis[iorb] = 1
            jcfg = lb.encode(tmp_basis)
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

    tmp_basis = []*rb.num_orbitals
    for lorb, korb, jorb, iorb in nonzero:
        if iorb == jorb or korb == lorb:
            continue
        for icfg in range(rb.dim):
            tmp_basis[:] = rb.decode(icfg)
            if tmp_basis[iorb] == 0:
                continue
            else:
                s1 = (-1)**np.count_nonzero(tmp_basis[0:iorb])
                tmp_basis[iorb] = 0
            if tmp_basis[jorb] == 0:
                continue
            else:
                s2 = (-1)**np.count_nonzero(tmp_basis[0:jorb])
                tmp_basis[jorb] = 0
            if tmp_basis[korb] == 1:
                continue
            else:
                s3 = (-1)**np.count_nonzero(tmp_basis[0:korb])
                tmp_basis[korb] = 1
            if tmp_basis[lorb] == 1:
                continue
            else:
                s4 = (-1)**np.count_nonzero(tmp_basis[0:lorb])
                tmp_basis[lorb] = 1
            jcfg = lb.encode(tmp_basis)
            if jcfg != -1:
                 rows.append(jcfg)
                 cols.append(icfg)
                 data.append(umat[lorb, korb, jorb, iorb] * s1 * s2 * s3 * s4)

    # We want to interface between the functions using csr. Eventually, we need to generate 
    # these directly, for now, let scipy convert.
    A = coo_matrix((data, (rows, cols)), shape=(lb.dim, rb.dim), dtype=np.complex128).tocsr()
    return A.indptr, A.indices, A.data, lb.dim, rb.dim


from petsc4py import PETSc

def get_H_emat(emat, lb, rb=None):
    indptr, indices, data, nl, nr = two_fermion_B(emat, lb, rb)
    return PETSc.Mat().createAIJ(size=(nl, nr), csr=(indptr, indices, data))


def get_H_umat(umat, lb, rb=None):
    indptr, indices, data, nl, nr = four_fermion_B(umat, lb, rb)
    return PETSc.Mat().createAIJ(size=(nl, nr), csr=(indptr, indices, data))


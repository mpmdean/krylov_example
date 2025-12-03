import numpy as np
import scipy

from edrixs.angular_momentum import (
    get_sx, get_sy, get_sz, get_lx, get_ly, get_lz, rmat_to_euler, get_wigner_dmat
)
from edrixs.photon_transition import (
    get_trans_oper, quadrupole_polvec, dipole_polvec_xas, dipole_polvec_rixs, unit_wavevector
)
from edrixs.coulomb_utensor import get_umat_slater, get_umat_slater_3shells
from edrixs.manybody_operator import two_fermion, four_fermion
from edrixs.fock_basis import get_fock_bin_by_N, write_fock_dec_by_N
from edrixs.basis_transform import cb_op2, tmat_r2c, cb_op
from edrixs.utils import info_atomic_shell, slater_integrals_name, boltz_dist
from edrixs.rixs_utils import scattering_mat
# from .plot_spectrum import get_spectra_from_poles, merge_pole_dicts
from edrixs.soc import atom_hsoc


def ed_1v1c_py(shell_name, *, shell_level=None, v_soc=None, c_soc=0,
               v_noccu=1, slater=None, ext_B=None, on_which='spin',
               v_cfmat=None, v_othermat=None, loc_axis=None, verbose=0):
    """
    Perform ED for the case of two atomic shells, one valence plus one Core
    shell with pure Python solver.
    For example, for Ni-:math:`L_3` edge RIXS, they are 3d valence and 2p core shells.

    It will use scipy.linalag.eigh to exactly diagonalize both the initial and intermediate
    Hamiltonians to get all the eigenvalues and eigenvectors, and the transition operators
    will be built in the many-body eigenvector basis.

    This solver is only suitable for small size of Hamiltonian, typically the dimension
    of both initial and intermediate Hamiltonian are smaller than 10,000.

    Parameters
    ----------
    shell_name: tuple of two strings
        Names of valence and core shells. The 1st (2nd) string in the tuple is for the
        valence (core) shell.

        - The 1st string can only be 's', 'p', 't2g', 'd', 'f',

        - The 2nd string can be 's', 'p', 'p12', 'p32', 'd', 'd32', 'd52',
          'f', 'f52', 'f72'.

        For example: shell_name=('d', 'p32') indicates a :math:`L_3` edge transition from
        core :math:`p_{3/2}` shell to valence :math:`d` shell.
    shell_level: tuple of two float numbers
        Energy level of valence (1st element) and core (2nd element) shells.

        They will be set to zero if not provided.
    v_soc: tuple of two float numbers
        Spin-orbit coupling strength of valence electrons, for the initial (1st element)
        and intermediate (2nd element) Hamiltonians.

        They will be set to zero if not provided.
    c_soc: a float number
        Spin-orbit coupling strength of core electrons.
    v_noccu: int number
        Number of electrons in valence shell.
    slater: tuple of two lists
        Slater integrals for initial (1st list) and intermediate (2nd list) Hamiltonians.
        The order of the elements in each list should be like this:

        [FX_vv, FX_vc, GX_vc, FX_cc],

        where X are integers with ascending order, it can be X=0, 2, 4, 6 or X=1, 3, 5.
        One can ignore all the continuous zeros at the end of the list.

        For example, if the full list is: [F0_dd, F2_dd, F4_dd, 0, F2_dp, 0, 0, 0, 0], one can
        just provide [F0_dd, F2_dd, F4_dd, 0, F2_dp]

        All the Slater integrals will be set to zero if slater=None.
    ext_B: tuple of three float numbers
        Vector of external magnetic field with respect to global :math:`xyz`-axis.

        They will be set to zero if not provided.
    on_which: string
        Apply Zeeman exchange field on which sector. Options are 'spin', 'orbital' or 'both'.
    v_cfmat: 2d complex array
        Crystal field splitting Hamiltonian of valence electrons. The dimension and the orbital
        order should be consistent with the type of valence shell.

        They will be zeros if not provided.
    v_othermat: 2d complex array
        Other possible Hamiltonian of valence electrons. The dimension and the orbital order
        should be consistent with the type of valence shell.

        They will be zeros if not provided.
    loc_axis: 3*3 float array
        The local axis with respect to which local orbitals are defined.

        - x: local_axis[:,0],

        - y: local_axis[:,1],

        - z: local_axis[:,2].

        It will be an identity matrix if not provided.
    verbose: int
        Level of writting data to files. Hopping matrices, Coulomb tensors, eigvenvalues
        will be written if verbose > 0.

    Returns
    -------
    eval_i:  1d float array
        The eigenvalues of initial Hamiltonian.
    eval_n: 1d float array
        The eigenvalues of intermediate Hamiltonian.
    trans_op: 3d complex array
        The matrices of transition operators in the eigenvector basis.
        Their components are defined with respect to the global :math:`xyz`-axis.
    """
    if verbose > 0:
        print("edrixs >>> Running ED ...")
    v_name_options = ['s', 'p', 't2g', 'd', 'f']
    c_name_options = ['s', 'p', 'p12', 'p32', 't2g', 'd', 'd32', 'd52', 'f', 'f52', 'f72']
    v_name = shell_name[0].strip()
    c_name = shell_name[1].strip()
    if v_name not in v_name_options:
        raise Exception("NOT supported type of valence shell: ", v_name)
    if c_name not in c_name_options:
        raise Exception("NOT supported type of core shell: ", c_name)

    info_shell = info_atomic_shell()

    # Quantum numbers of angular momentum
    v_orbl = info_shell[v_name][0]

    # number of orbitals including spin degree of freedom
    v_norb = info_shell[v_name][1]
    c_norb = info_shell[c_name][1]

    # total number of orbitals
    ntot = v_norb + c_norb

    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)

    # Coulomb interaction
    # Get the names of all the required slater integrals
    slater_name = slater_integrals_name((v_name, c_name), ('v', 'c'))
    nslat = len(slater_name)

    slater_i = np.zeros(nslat, dtype=float)
    slater_n = np.zeros(nslat, dtype=float)

    if slater is not None:
        if nslat > len(slater[0]):
            slater_i[0:len(slater[0])] = slater[0]
        else:
            slater_i[:] = slater[0][0:nslat]
        if nslat > len(slater[1]):
            slater_n[0:len(slater[1])] = slater[1]
        else:
            slater_n[:] = slater[1][0:nslat]

    if verbose > 0:
        # print summary of slater integrals
        print()
        print("    Summary of Slater integrals:")
        print("    ------------------------------")
        print("    Terms,   Initial Hamiltonian,  Intermediate Hamiltonian")
        for i in range(nslat):
            print("    ", slater_name[i], ":  {:20.10f}{:20.10f}".format(slater_i[i], slater_n[i]))
        print()

    case = v_name + c_name
    umat_i = get_umat_slater(case, *slater_i)
    umat_n = get_umat_slater(case, *slater_n)

    if verbose > 0:
        write_umat(umat_i, 'coulomb_i.in')
        write_umat(umat_n, 'coulomb_n.in')

    # SOC
    if v_soc is not None:
        emat_i[0:v_norb, 0:v_norb] += atom_hsoc(v_name, v_soc[0])
        emat_n[0:v_norb, 0:v_norb] += atom_hsoc(v_name, v_soc[1])

    # when the core-shell is any of p12, p32, d32, d52, f52, f72,
    # do not need to add SOC for core shell
    if c_name in ['p', 'd', 'f']:
        emat_n[v_norb:ntot, v_norb:ntot] += atom_hsoc(c_name, c_soc)

    # crystal field
    if v_cfmat is not None:
        emat_i[0:v_norb, 0:v_norb] += np.array(v_cfmat)
        emat_n[0:v_norb, 0:v_norb] += np.array(v_cfmat)

    # other hopping matrix
    if v_othermat is not None:
        emat_i[0:v_norb, 0:v_norb] += np.array(v_othermat)
        emat_n[0:v_norb, 0:v_norb] += np.array(v_othermat)

    # energy of shells
    if shell_level is not None:
        emat_i[0:v_norb, 0:v_norb] += np.eye(v_norb) * shell_level[0]
        emat_i[v_norb:ntot, v_norb:ntot] += np.eye(c_norb) * shell_level[1]
        emat_n[0:v_norb, 0:v_norb] += np.eye(v_norb) * shell_level[0]
        emat_n[v_norb:ntot, v_norb:ntot] += np.eye(c_norb) * shell_level[1]

    # external magnetic field
    if v_name == 't2g':
        lx, ly, lz = get_lx(1, True), get_ly(1, True), get_lz(1, True)
        sx, sy, sz = get_sx(1), get_sy(1), get_sz(1)
        lx, ly, lz = -lx, -ly, -lz
    else:
        lx, ly, lz = get_lx(v_orbl, True), get_ly(v_orbl, True), get_lz(v_orbl, True)
        sx, sy, sz = get_sx(v_orbl), get_sy(v_orbl), get_sz(v_orbl)

    if ext_B is not None:
        if on_which.strip() == 'spin':
            zeeman = ext_B[0] * (2 * sx) + ext_B[1] * (2 * sy) + ext_B[2] * (2 * sz)
        elif on_which.strip() == 'orbital':
            zeeman = ext_B[0] * lx + ext_B[1] * ly + ext_B[2] * lz
        elif on_which.strip() == 'both':
            zeeman = ext_B[0] * (lx + 2 * sx) + ext_B[1] * (ly + 2 * sy) + ext_B[2] * (lz + 2 * sz)
        else:
            raise Exception("Unknown value of on_which", on_which)
        emat_i[0:v_norb, 0:v_norb] += zeeman
        emat_n[0:v_norb, 0:v_norb] += zeeman

    if verbose > 0:
        write_emat(emat_i, 'hopping_i.in')
        write_emat(emat_n, 'hopping_n.in')

    basis_i = get_fock_bin_by_N(v_norb, v_noccu, c_norb, c_norb)
    basis_n = get_fock_bin_by_N(v_norb, v_noccu+1, c_norb, c_norb - 1)
    ncfg_i, ncfg_n = len(basis_i), len(basis_n)
    if verbose > 0:
        print("edrixs >>> Dimension of the initial Hamiltonian: ", ncfg_i)
        print("edrixs >>> Dimension of the intermediate Hamiltonian: ", ncfg_n)
        # Build many-body Hamiltonian in Fock basis
        print("edrixs >>> Building Many-body Hamiltonians ...")
    
    hmat_i = np.zeros((ncfg_i, ncfg_i), dtype=complex)
    hmat_n = np.zeros((ncfg_n, ncfg_n), dtype=complex)
    hmat_i[:, :] += two_fermion(emat_i, basis_i, basis_i)
    hmat_i[:, :] += four_fermion(umat_i, basis_i)
    hmat_n[:, :] += two_fermion(emat_n, basis_n, basis_n)
    hmat_n[:, :] += four_fermion(umat_n, basis_n)
    if verbose > 0:
        print("edrixs >>> Done !")

    # Do exact-diagonalization to get eigenvalues and eigenvectors
    if verbose > 0:
        print("edrixs >>> Exact Diagonalization of Hamiltonians ...")
    eval_i, evec_i = scipy.linalg.eigh(hmat_i)
    eval_n, evec_n = scipy.linalg.eigh(hmat_n)
    if verbose > 0:
        print("edrixs >>> Done !")

    if verbose > 0:
        write_tensor(eval_i, 'eval_i.dat')
        write_tensor(eval_n, 'eval_n.dat')

    # Build dipolar transition operators in local-xyz axis
    if loc_axis is not None:
        local_axis = np.array(loc_axis)
    else:
        local_axis = np.eye(3)
    tmp = get_trans_oper(case)
    npol, n, m = tmp.shape
    tmp_g = np.zeros((npol, n, m), dtype=complex)
    # Transform the transition operators to global-xyz axis
    # dipolar transition
    if npol == 3:
        for i in range(3):
            for j in range(3):
                tmp_g[i] += local_axis[i, j] * tmp[j]

    # quadrupolar transition
    elif npol == 5:
        alpha, beta, gamma = rmat_to_euler(local_axis)
        wignerD = get_wigner_dmat(4, alpha, beta, gamma)
        rotmat = np.dot(np.dot(tmat_r2c('d'), wignerD), np.conj(np.transpose(tmat_r2c('d'))))
        for i in range(5):
            for j in range(5):
                tmp_g[i] += rotmat[i, j] * tmp[j]
    else:
        raise Exception("Have NOT implemented this case: ", npol)

    tmp2 = np.zeros((npol, ntot, ntot), dtype=complex)
    trans_op = np.zeros((npol, ncfg_n, ncfg_i), dtype=complex)
    for i in range(npol):
        tmp2[i, 0:v_norb, v_norb:ntot] = tmp_g[i]
        trans_op[i] = cb_op2(trans_op[i], evec_n, evec_i)

    if verbose > 0:
        print("edrixs >>> ED Done !")

    hmat_i = scipy.sparse.csr_matrix(hmat_i)
    hmat_n = scipy.sparse.csr_matrix(hmat_n)

    return eval_i, evec_i, eval_n, evec_n, trans_op, basis_i, basis_n, hmat_i, hmat_n, ntot




def rixs_1v1c_py(eval_i, eval_n, trans_op, ominc, eloss, *,
                 gamma_c=0.1, gamma_f=0.01, thin=1.0, thout=1.0, phi=0.0,
                 pol_type=None, gs_list=None, temperature=1.0, scatter_axis=None, skip_gs=False, verbose=0):
    """
    Calculate RIXS for the case of one valence shell plus one core shell with Python solver.

    This solver is only suitable for small size of Hamiltonian, typically the dimension
    of both initial and intermediate Hamiltonian are smaller than 10,000.

    Parameters
    ----------
    eval_i: 1d float array
        The eigenvalues of the initial Hamiltonian.
    eval_n: 1d float array
        The eigenvalues of the intermediate Hamiltonian.
    trans_op: 3d complex array
        The transition operators in the eigenstates basis.
    ominc: 1d float array
        Incident energy of photon.
    eloss: 1d float array
        Energy loss.
    gamma_c: a float number or a 1d float array with same shape as ominc.
        The core-hole life-time broadening factor. It can be a constant value
        or incident energy dependent.
    gamma_f: a float number or a 1d float array with same shape as eloss.
        The final states life-time broadening factor. It can be a constant value
        or energy loss dependent.
    thin: float number
        The incident angle of photon (in radian).
    thout: float number
        The scattered angle of photon (in radian).
    phi: float number
        Azimuthal angle (in radian), defined with respect to the
        :math:`x`-axis of scattering axis: scatter_axis[:,0].
    pol_type: list of 4-elements-tuples
        Type of polarizations. It has the following form:

        (str1, alpha, str2, beta)

        where, str1 (str2) can be 'linear', 'left', 'right', 'isotropic' and alpha (beta) is
        the angle (in radians) between the linear polarization vector and the scattering plane.

        If str1 (or str2) is 'isotropic' then the polarization vector projects equally
        along each axis and the other variables are ignored.

        It will set pol_type=[('linear', 0, 'linear', 0)] if not provided.
    gs_list: 1d list of ints
        The indices of initial states which will be used in RIXS calculations.

        It will set gs_list=[0] if not provided.
    temperature: float number
        Temperature (in K) for boltzmann distribution.
    scatter_axis: 3*3 float array
        The local axis defining the scattering plane. The scattering plane is defined in
        the local :math:`zx`-plane.

        - local :math:`x`-axis: scatter_axis[:,0]

        - local :math:`y`-axis: scatter_axis[:,1]

        - local :math:`z`-axis: scatter_axis[:,2]

        It will be an identity matrix if not provided.
    skip_gs: bool
        If True, transitions to the ground state(s) (forming the elastic peak) are omitted from
        the calculation.
    verbose: bool
        If true print (default false)

    Returns
    -------
    rixs: 3d float array
        The calculated RIXS spectra. The 1st dimension is for the incident energy,
        the 2nd dimension is for the energy loss and the 3rd dimension is for
        different polarizations.
    """

    if verbose > 0:
        print("edrixs >>> Running RIXS ... ")
    n_ominc = len(ominc)
    n_eloss = len(eloss)
    gamma_core = np.zeros(n_ominc, dtype=float)
    gamma_final = np.zeros(n_eloss, dtype=float)
    if np.isscalar(gamma_c):
        gamma_core[:] = np.ones(n_ominc) * gamma_c
    else:
        gamma_core[:] = gamma_c

    if np.isscalar(gamma_f):
        gamma_final[:] = np.ones(n_eloss) * gamma_f
    else:
        gamma_final[:] = gamma_f

    if pol_type is None:
        pol_type = [('linear', 0, 'linear', 0)]
    if gs_list is None:
        gs_list = [0]
    if scatter_axis is None:
        scatter_axis = np.eye(3)
    else:
        scatter_axis = np.array(scatter_axis)

    prob = boltz_dist([eval_i[i] for i in gs_list], temperature)
    rixs = np.zeros((len(ominc), len(eloss), len(pol_type)), dtype=float)
    npol, n, m = trans_op.shape
    trans_emi = np.zeros((npol, m, n), dtype=np.complex128)
    for i in range(npol):
        trans_emi[i] = np.conj(np.transpose(trans_op[i]))
    polvec_i = np.zeros(npol, dtype=complex)
    polvec_f = np.zeros(npol, dtype=complex)

    # Calculate RIXS
    for i, om in enumerate(ominc):
        F_fi = scattering_mat(eval_i, eval_n, trans_op[:, :, 0:max(gs_list)+1],
                              trans_emi, om, gamma_core[i])

        for j, (it, alpha, jt, beta) in enumerate(pol_type):
            ei, ef = dipole_polvec_rixs(thin, thout, phi, alpha, beta,
                                        scatter_axis, (it, jt))
            if it.lower() == 'isotropic':
                ei = np.ones(3)/np.sqrt(3)                        # Powder spectrum
            if jt.lower() == 'isotropic':
                ef = np.ones(3)/np.sqrt(3)
            # dipolar transition
            if npol == 3:
                polvec_i[:] = ei
                polvec_f[:] = ef
            # quadrupolar transition
            elif npol == 5:
                ki = unit_wavevector(thin, phi, scatter_axis, direction='in')
                kf = unit_wavevector(thout, phi, scatter_axis, direction='out')
                polvec_i[:] = quadrupole_polvec(ei, ki)
                polvec_f[:] = quadrupole_polvec(ef, kf)
            else:
                raise Exception("Have NOT implemented this type of transition operators")
            # scattering magnitude with polarization vectors
            F_mag = np.zeros((len(eval_i), len(gs_list)), dtype=complex)
            for m in range(npol):
                for n in range(npol):
                    F_mag[:, :] += np.conj(polvec_f[m]) * F_fi[m, n] * polvec_i[n]

            fs_list = np.arange(len(eval_i))
            if skip_gs:
                fs_list = np.delete(fs_list, gs_list)
            for m, igs in enumerate(gs_list):
                for n in fs_list:
                    rixs[i, :, j] += (
                        prob[m] * np.abs(F_mag[n, igs])**2 * gamma_final / np.pi /
                        ((eloss - (eval_i[n] - eval_i[igs]))**2 + gamma_final**2)
                    )
    if verbose > 0:
        print("edrixs >>> RIXS Done !")

    return rixs

# coding: utf-8
"""
Created on 16 Jan 2020

project: LatticeQMC
version: 1.0
"""
import os
import numpy as np


def _get_filepath(model, beta, time_steps, sweeps):
    """ Creates the file path for the given parameters"""
    w, h = model.lattice.shape
    folder = os.path.join("data", f"u={model.u}_t={model.t}_mu={model.mu}_w={w}_h={h}")
    file = f"gf_itau__beta={beta}_nt={time_steps}_sweeps={sweeps}.npy"
    return os.path.abspath(os.path.join(folder, file))


def save_gf_tau(model, beta, time_steps, sweeps, gf):
    """ Save imaginary time depended Green's function measurement data into the tree """
    file = _get_filepath(model, beta, time_steps, sweeps)
    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    np.save(file, gf)


def load_gf_tau(model, beta, time_steps, sweeps):
    """ Load imaginary time depended Green's function measurement data """
    file = _get_filepath(model, beta, time_steps, sweeps)
    return np.load(file)


def check_params(u, t, dtau):
    r""" Checks the configuration of the model and HS-field.

    .. math::
        U t (\Delta\tau)^2 < \frac{1}{10}

    Parameters
    ----------
    u: float
        Hubbard interaction :math:`U`.
    t: float
        Hopping parameter :math:'t'.
    dtau: float
        Time slice size of the HS-field.
    """
    check_val = u * t * dtau**2
    if check_val < 0.1:
        print(f"Check-value {check_val:.2} is smaller than 0.1!")
    else:
        print(f"Check-value {check_val:.2} should be smaller than 0.1!")


def filling(g_sigma):
    r""" Computes the local filling of the model.

    Parameters
    ----------
    g_sigma: (N, N) np.ndarray
        Green's function .math'G_{\sigma}' of a spin channel.

    Returns
    -------
    n: (N) np.ndarray
    """
    return 1 - np.diagonal(g_sigma)


def print_filling(gf_up, gf_dn):
    n_up = filling(gf_up)
    n_dn = filling(gf_dn)
    print(f"<n↑> = {np.mean(n_up):.3f}  {n_up}")
    print(f"<n↓> = {np.mean(n_dn):.3f}  {n_dn}")
    print(f"<n>  = {np.mean(n_up + n_dn):.3f}")


def local_gf(gf):
    """ Returns the local elements (diagonal) of the Green's function matrix

    Parameters
    ----------
    gf: (..., N, N) array_like
        Green's function matrice

    Returns
    -------
    gf_loc: (..., N) np.ndarray
    """
    return np.diagonal(gf, axis1=-2, axis2=-1)


def matsubara_frequencies(points, beta):
    """ Returns the fermionic Matsubara frequencies :math:'iω_n' for the points 'points'.

    Parameters
    ----------
    points: array_like
        Points for which the Matsubara frequencies :math:`iω_n` are returned.
    beta: float
        The inverse temperature :math:'beta = 1/k_B T'.

    Returns
    -------
    matsubara_frequencies: complex ndarray
    """
    n_points = np.asanyarray(points).astype(dtype=int, casting='safe')
    return 1j * np.pi / beta * (2 * n_points + 1)


def fermi_fct(eps, beta):
    r"""Return the Fermi function '1/(exp(βϵ)+1)'.

    For complex inputs the function is not as accurate as for real inputs.

    Parameters
    ----------
    eps : complex or float or ndarray
        The energy at which the Fermi function is evaluated.
    beta : float
        The inverse temperature :math:'beta = 1/(k_B T)'.

    Returns
    -------
    fermi_fct : complex of float or ndarray
        The Fermi function, same type as eps.
    """
    return 0.5 * (1. + np.tanh(-0.5 * beta * eps))

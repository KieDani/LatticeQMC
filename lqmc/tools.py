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


# =========================================================================
#                               GF TOOLS
# =========================================================================


def decompose(a):
    xi, rv = np.linalg.eigh(a)
    return rv, xi, np.linalg.inv(rv)


def reconstruct(rv, xi, rv_inv, diag=False):
    if diag:
        return ((np.transpose(rv_inv) * rv) @ xi[..., np.newaxis])[..., 0]
    else:
        return (rv * xi[..., np.newaxis, :]) @ rv_inv


def pole_gf_tau(tau, poles, weights, beta):
    assert np.all((tau >= 0.) & (tau <= beta))
    poles, weights = np.atleast_1d(*np.broadcast_arrays(poles, weights))
    tau = np.asanyarray(tau)
    tau = tau.reshape(tau.shape + (1,)*poles.ndim)
    # exp(-tau*pole)*f(-pole, beta) = exp((beta-tau)*pole)*f(pole, beta)
    exponent = np.where(poles.real >= 0, -tau, beta-tau) * poles
    single_pole_tau = np.exp(exponent) * fermi_fct(-np.sign(poles.real)*poles, beta)
    return -np.sum(weights*single_pole_tau, axis=-1)


def compute_pole_gf_tau(ham, beta):
    rv, xi, rv_inv = decompose(ham)
    tau = np.linspace(0, beta, num=2049)
    # append axis, as we don't want the sum here
    diag_gf_tau = pole_gf_tau(tau, xi[..., np.newaxis], weights=1, beta=beta)
    gf_tau = reconstruct(rv, diag_gf_tau, rv_inv)
    gf_tau = np.moveaxis(gf_tau, 0, -1)  # Convert to shape (site, site, tau)
    return tau, gf_tau

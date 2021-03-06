# coding: utf-8
"""
Created on 16 Jan 2020

project: LatticeQMC
version: 1.0
"""
import os
import numpy as np


def get_datapath(filename, model, post='', mkdir=True, **kwargs):
    """ Creates the file path for the given parameters

    Parameters
    ----------
    filename: str
        Name of the file.
    model: HubbardModel
        Instance of the 'HubbardModel'-object used.
    post: str, optional
        Optional string to append to the final path (before the file extension).
        This is used for creating tmp-files (<name>_tmp.ext)
    mkdir: bool, optional
        Flag if the directory path of the file should be created. The default is True.
    **kwargs
        Optional keyword arguments of parameters used in the filepath.

    Returns
    -------
    path: str
    """
    w, h = model.lattice.shape
    folder = os.path.join("data", f"u={model.u}_t={model.t}_mu={model.mu}_w={w}_h={h}")
    if mkdir and not os.path.isdir(folder):
        os.makedirs(folder)
    kwargstr = '_'.join([f'{key}={val}' for key, val in kwargs.items()])
    return os.path.abspath(os.path.join(folder, f"{filename} {kwargstr}{post}.npz"))


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


def filling(g_sigma, site=None, axis1=-2, axis2=-1):
    r""" Computes the local filling of the model.

    Parameters
    ----------
    g_sigma: (..., N, N) np.ndarray
        Green's function .math'G_{\sigma}' of a spin channel.
    site: int, optional
        Site index. If not given the whole array is returned.
    axis1: int, optional
        The first axis of the gf-matrix. By default the last two axes are used.
    axis2: int, optional
        The second axis of the gf-matrix. By default the last two axes are used.

    Returns
    -------
    n: (..., N) np.ndarray
    """
    n = 1 - np.diagonal(g_sigma, axis1=axis1, axis2=axis2)
    if site is not None:
        n = n[..., site]
    return n


def local_moment(gf_up, gf_dn, site=None, axis1=-2, axis2=-1):
    r""" Computes the local moment of the model.

    Parameters
    ----------
    gf_up: (..., N, N) np.ndarray
        Spin-up Green's function .math'G_{\uparrow}'.
    gf_dn: (..., N, N) np.ndarray
        Spin-down Green's function .math'G_{\downarrow}'.
    site: int, optional
        Site index. If not given the whole array is returned.
    axis1: int, optional
        The first axis of the gf-matrix. By default the last two axes are used.
    axis2: int, optional
        The second axis of the gf-matrix. By default the last two axes are used.

    Returns
    -------
    m: float or np.ndarray
    """
    n_up = filling(gf_up, site, axis1, axis2)
    n_dn = filling(gf_dn, site, axis1, axis2)
    return n_up + n_dn - 2 * n_up * n_dn


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

# coding: utf-8
"""
Created on 15 Jan 2020

project: LatticeQMC
version: 1.0

To Do
-----
- Efficiency
- Green's function time slices
"""
import logging
import itertools
import numpy as np
from scipy.linalg import expm
from lqmc import HubbardModel, Configuration

# Configure basic logging for lqmc-loop
log_file = 'data\\lqmc.log'
logging.basicConfig(filename=log_file, filemode="w", format='%(message)s', level=logging.DEBUG)


def updateln(string):
    """ Rewrites the current console line

    Parameters
    ----------
    string: str
        String to display.
    """
    print("\r" + string, end="", flush=True)


def compute_lambda(u, dtau):
    r""" Computes the factor '\lambda' used in the computation of 'M'.

    Parameters
    ----------
    u: float
        Hubbard interaction :math:`U`.
    dtau: float
        Time slice size of the HS-field.

    Returns
    -------
    lamb: float
    """
    return np.arccosh(np.exp(u * dtau / 2.))


def compute_m(ham_kin, config, lamb, dtau, sigma):
    r""" Computes the matrix :math:'M' used in the Metropolis ratio

    The matrix :math:'M' is defined as
    .. math::
       M_\sigma(h) = I + B_{L, \sigma}(h_{L}) B_{L-1, \sigma}(h_{L-1}) \dots B_{1, \sigma}(h_{1})

    with
    .. math::
       B_{l, \sigma}(h_l) = e^{\Delta\tau K} e^{\sigma \lambda V_l(h_l)}

    .math'V_l(h_l) = diag(h_{1l}, \dots h_{Nl})' is the diagonal matrix of a time slice
    of the HS-field .math'h_{il}'

    Parameters
    ----------
    ham_kin: (N, N) np.ndarray
        Kinetic hamiltonian .math'K'.
    config: Configuration
        The current HS-field configuration object.
        This contains the field .math'h_{il}'
    lamb: float
        Factor .math'\lambda'.
    dtau: float
        Time slice size of the HS-field.
    sigma: int
        Spin index.

    Returns
    -------
    m: (N, N) np.ndarray
    """
    n = ham_kin.shape[0]

    # Calculate the first matrix exp of B. This is a static value.
    # check if there is a better way to calculate the matrix exponential
    exp_k = expm(dtau * ham_kin)
    # Create the V_l matrix
    v = np.zeros((n, n), dtype=config.dtype)

    # fill diag(V_l) with values of last time slice and compute B-product
    lmax = config.n_t - 1

    np.fill_diagonal(v, config[:, lmax])
    exp_v = expm(sigma * lamb * v)
    b = np.dot(exp_k, exp_v)

    b_prod = b
    for l in reversed(range(1, lmax)):
        # Fill V_l with new values, compute B(l) and multiply with total product
        np.fill_diagonal(v, config[:, l])
        exp_v = expm(sigma * lamb * v)
        b = np.dot(exp_k, exp_v)
        b_prod = np.dot(b_prod, b)

    # compute M matrix
    return np.eye(n) + b_prod


def compute_gf_tau(config, ham_kin, g_beta, lamb, dtau, sigma):
    r""" Computes the Green's function for all time slices

    Parameters
    ----------
    config: Configuration
        The current HS-field configuration object.
        This contains the field .math'h_{il}'
    ham_kin: (N, N) np.ndarray
        Kinetic hamiltonian .math'K'.
    g_beta: (N, N) np.ndarray
        The Green's function matrix for all spatial sites N at time .math'\beta'
    lamb: float
        Factor .math'\lambda'.
    dtau: float
        Time slice size of the HS-field.
    sigma: int
        Spin index.

    Returns
    -------
    gf: (M, N, N) np.ndarray
        The Green's function for N sites and M time slices.
    """
    n = ham_kin.shape[0]

    # Calculate the first matrix exp of B. This is a static value.
    # check if there is a better way to calculate the matrix exponential
    exp_k = expm(dtau * ham_kin)

    v = np.zeros((n, n), dtype=config.dtype)

    # g[0, :, :] is Greensfunction at time beta, g[1, :, :] is Greensfunction one step before, etc
    g = np.zeros((config.n_t, n, n), dtype=np.float64)
    g[0, :, :] = g_beta
    for l in range(1, config.n_t):
        # Create the V_l matrix
        np.fill_diagonal(v, config[:, l])
        exp_v = expm(sigma * lamb * v)

        b = np.dot(exp_k, exp_v)
        g[l, :, :] = np.dot(np.dot(b, g[l-1, :, :]), np.linalg.inv(b))
    return g


def warmup_loop(model, config, dtau, sweeps=200, fast=True):
    """ Runs the warmup lqmc-loop

    Parameters
    ----------
    model: HubbardModel
        The Hubbard model instance.
    config: Configuration
        The current HS-field configuration object after warmup.
        This contains the field .math'h_{il}'
    dtau: float
        Time slice size of the HS-field.
    sweeps: int, optional
        Number of sweeps through the HS-field.
    fast: bool, optional
        Flag if the fast algorithm should be used. The default is True.

    Returns
    -------
    config: Configuration
        The updated HS-field configuration after the warmup loop.
    """
    ham_kin = model.ham_kinetic()

    # Calculate factor
    lamb = compute_lambda(model.u, dtau)

    # Calculate m-matrices
    m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Store copy of current configuration
    old_config = config.copy()
    # QMC loop
    acc = False
    ratio = 0
    updateln("Warmup sweep")
    for sweep in range(sweeps):
        for i, l in itertools.product(range(model.n_sites), range(config.n_t)):
            updateln(f"Warmup sweep: {sweep+1}/{sweeps}, accepted: {acc} (ratio={ratio:.2f})")
            if fast:
                # Calculate m-matrices and ratio of the configurations
                # Accept move with metropolis acceptance ratio.
                m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
                m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
                d_up = 1 + (1 - np.linalg.inv(m_up)[i, i]) * (np.exp(-2 * lamb * config[i, l]) - 1)
                d_dn = 1 + (1 - np.linalg.inv(m_dn)[i, i]) * (np.exp(+2 * lamb * config[i, l]) - 1)
                ratio = d_up * d_dn
                r = np.random.rand()  # Random number between 0 and 1
                if r < ratio:
                    # Move accepted:
                    # Update configuration
                    acc = True
                    config.update(i, l)
                else:
                    # Move not accepted!
                    acc = False
            else:
                # Update Configuration
                config.update(i, l)
                # Calculate m-matrices and ratio of the configurations
                # Accept move with metropolis acceptance ratio.
                m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
                m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
                new = np.linalg.det(m_up) * np.linalg.det(m_dn)
                ratio = new / old
                r = np.random.rand()  # Random number between 0 and 1
                if r < ratio:
                    # Move accepted:
                    # Continue using the new configuration
                    acc = True
                    old = new
                    old_config = config.copy()
                else:
                    # Move not accepted:
                    # Revert to the old configuration
                    acc = False
                    config = old_config

            logging.info(f"[Warmup] Sweep={sweep} i={i}, l={l} - ratio={ratio:.3f}, accepted={acc}")
    print()
    return config


def measure_loop(model, config, dtau, sweeps=800, fast=True):
    r""" Runs the measurement lqmc-loop and returns the measured Green's function

    Parameters
    ----------
    model: HubbardModel
        The Hubbard model instance.
    config: Configuration
        The current HS-field configuration object.
        This contains the field .math'h_{il}'
    dtau: float
        Time slice size of the HS-field.
    sweeps: int, optional
        Number of sweeps through the HS-field.
    fast: bool, optional
        Flag if the fast algorithm should be used. The default is True.

    Returns
    -------
    gf_dn: (M, N, N) np.ndarray
        Measured spin-up Green's function .math'G_\uparrow(\tau)' for all time slices.
    gf_dn: (M, N, N) np.ndarray
        Measured spin-down Green's function .math'G_\downarrow(\tau)' for all time slices.
    """
    ham_kin = model.ham_kinetic()
    n = model.n_sites

    # Calculate factor
    lamb = compute_lambda(model.u, dtau)

    # Calculate m-matrices
    m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Initialize total and temp greens functions
    gf_up, gf_dn = 0, 0
    g_beta_up = np.linalg.inv(m_up)
    g_beta_dn = np.linalg.inv(m_dn)

    # Compute Greens function for all time steps using the gf at time beta
    g_tmp_up = compute_gf_tau(config, ham_kin, g_beta_up, lamb, dtau, sigma=+1)
    g_tmp_dn = compute_gf_tau(config, ham_kin, g_beta_dn, lamb, dtau, sigma=-1)

    # Store copy of current configuration
    old_config = config.copy()

    # QMC loop
    acc = False
    ratio = 0
    number = 0
    updateln("Measurement sweep")
    for sweep in range(sweeps):
        for i, l in itertools.product(range(model.n_sites), range(config.n_t)):
            updateln(f"Measurement sweep: {sweep+1}/{sweeps}, accepted: {acc} (ratio={ratio:.2f})")

            if fast:
                # Calculate m-matrices and ratio of the configurations
                # Accept move with metropolis acceptance ratio.
                m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
                m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
                d_up = 1 + (1 - np.linalg.inv(m_up)[i, i]) * (np.exp(-2 * lamb * config[i, l]) - 1)
                d_dn = 1 + (1 - np.linalg.inv(m_dn)[i, i]) * (np.exp(+2 * lamb * config[i, l]) - 1)
                ratio = d_up * d_dn
                r = np.random.rand()  # Random number between 0 and 1
                if r < ratio:
                    # Move accepted:
                    # Update temp greens function and update configuration
                    c_up = np.zeros(n, dtype=np.float64)
                    c_dn = np.zeros(n, dtype=np.float64)
                    c_up[i] = np.exp(-2 * lamb * config[i, l]) - 1
                    c_dn[i] = np.exp(+2 * lamb * config[i, l]) - 1
                    c_up = -1 * (np.exp(-2 * lamb * config[i, l]) - 1) * g_beta_up[i, :] + c_up
                    c_dn = -1 * (np.exp(+2 * lamb * config[i, l]) - 1) * g_beta_dn[i, :] + c_dn

                    b_up = g_beta_up[:, i] / (1. + c_up[i])
                    b_dn = g_beta_dn[:, i] / (1. + c_dn[i])

                    g_beta_up = g_beta_up - np.outer(b_up, c_up)
                    g_beta_dn = g_beta_dn - np.outer(b_dn, c_dn)
                    g_tmp_up = compute_gf_tau(config, ham_kin, g_beta_up, lamb, dtau, sigma=+1)
                    g_tmp_dn = compute_gf_tau(config, ham_kin, g_beta_dn, lamb, dtau, sigma=-1)

                    acc = True
                    # Update Configuration
                    config.update(i, l)
                else:
                    # Move not accepted:
                    # Revert to the old configuration
                    acc = False
            else:
                # Update Configuration
                config.update(i, l)
                # Calculate m-matrices and ratio of the configurations
                # Accept move with metropolis acceptance ratio.
                m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
                m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
                new = np.linalg.det(m_up) * np.linalg.det(m_dn)
                ratio = new / old
                r = np.random.rand()  # Random number between 0 and 1
                if r < ratio:
                    # Move accepted:
                    # Update temp greens function and continue using the new configuration
                    g_beta_up = np.linalg.inv(m_up)
                    g_beta_dn = np.linalg.inv(m_dn)
                    g_tmp_up = compute_gf_tau(config, ham_kin, g_beta_up, dtau, lamb, sigma=+1)
                    g_tmp_dn = compute_gf_tau(config, ham_kin, g_beta_dn, dtau, lamb, sigma=-1)

                    acc = True
                    old = new
                    old_config = config.copy()
                else:
                    # Move not accepted:
                    # Revert to the old configuration
                    acc = False
                    config = old_config

            # Add temp greens function to total gf after each step
            gf_up += g_tmp_up
            gf_dn += g_tmp_dn
            number += 1

            logging.info(f"[Measure] Sweep={sweep} i={i}, l={l} - ratio={ratio:.3f}, accepted={acc}")
    print()
    # Return the normalized gfs for each spin
    return np.array([gf_up, gf_dn]) / number

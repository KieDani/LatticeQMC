# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0

To do
-----
- Multiproccesing
- M-matrix inner func (??)
"""
import time
import itertools
import numpy as np
from scipy.linalg import expm
from lqmc import HubbardModel, Configuration
import logging

# Configure basic logging for lqmc-loop
log_file = 'data\\lqmc_slow.log'
logging.basicConfig(filename=log_file, filemode="w", format='%(message)s', level=logging.DEBUG)


def updateln(string):
    """ Rewrites the current console line

    Parameters
    ----------
    string: str
        String to display.
    """
    print("\r" + string, end="", flush=True)


def check_params(u, t, dtau):
    r""" Checks the configuration of the model and HS-field.

    .. math::
        U t \Delta\tau < \frac{1}{10}

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
        print(f"Check-value {check_val:.2f} is smaller than 0.1!")
    else:
        print(f"Check-value {check_val:.2f} should be smaller than 0.1!")


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


def warmup(model, config, dtau, sweeps=200):
    """ Runs the warmup lqmc-loop

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

    Returns
    -------
    config: Configuration
        The updated HS-field configuration after the warmup loop.
    """
    ham_kin = model.ham_kinetic()

    # Calculate factor
    lamb = np.arccosh(np.exp(model.u * dtau / 2.))

    # Calculate m-matrices
    m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Store copy of current configuration
    old_config = config.copy()
    acc = False
    # QMC loop
    updateln("Warmup sweep")
    for sweep in range(sweeps):
        for i, l in itertools.product(range(model.n_sites), range(config.n_t)):
            updateln(f"Warmup sweep: {sweep+1}/{sweeps}, accepted: {acc}")
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


def measure_gf(model, config, dtau, time_steps, sweeps=800):
    """ Runs the measurement lqmc-loop and returns the measured Green's function

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

    Returns
    -------
    gf: (2, N) np.ndarray
        Measured Green's function .math'G' of the up- and down-spin channel.
    """
    ham_kin = model.ham_kinetic()

    # Calculate factor
    lamb = np.arccosh(np.exp(model.u * dtau / 2.))

    # Calculate m-matrices
    m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Initialize total and temp greens functions
    gf_up, gf_dn = 0, 0
    g_tmp_up_beta = np.linalg.inv(m_up)
    g_tmp_dn_beta = np.linalg.inv(m_dn)
    g_tmp_up = gf_tau(g_beta=g_tmp_up_beta, ham_kin=ham_kin, dtau=dtau, lamb=lamb, sigma=+1, config=config,
                      time_steps=time_steps)
    g_tmp_dn = gf_tau(g_beta=g_tmp_dn_beta, ham_kin=ham_kin, dtau=dtau, lamb=lamb, sigma=-1, config=config,
                      time_steps=time_steps)

    # Store copy of current configuration
    old_config = config.copy()

    # QMC loop
    acc = False
    number = 0
    updateln("Measurement sweep")
    for sweep in range(sweeps):
        for i, l in itertools.product(range(model.n_sites), range(config.n_t)):
            updateln(f"Measurement sweep: {sweep+1}/{sweeps}, accepted: {acc}")
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
                g_tmp_up_beta = np.linalg.inv(m_up)
                g_tmp_dn_beta = np.linalg.inv(m_dn)
                g_tmp_up = gf_tau(g_beta=g_tmp_up_beta, ham_kin=ham_kin, dtau=dtau, lamb=lamb, sigma=+1, config=config,
                                  time_steps=time_steps)
                g_tmp_dn = gf_tau(g_beta=g_tmp_dn_beta, ham_kin=ham_kin, dtau=dtau, lamb=lamb, sigma=-1, config=config,
                                  time_steps=time_steps)

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



def gf_tau(g_beta, ham_kin, dtau, lamb, sigma, config, time_steps):
    n = ham_kin.shape[0]

    # Calculate the first matrix exp of B. This is a static value.
    # check if there is a better way to calculate the matrix exponential
    exp_k = expm(dtau * ham_kin)

    #g[0][:,:] is Greensfunction at time beta, g[1][0:0] is Greensfunction one step before, etc
    g=np.array((time_steps,n,n), dtype=np.float64)
    g[0][:,:]= g_beta

    for l in range(1,time_steps):
        # Create the V_l matrix
        v = np.zeros((n, n), dtype=config.dtype)
        np.fill_diagonal(v, config[:, l])
        exp_v = expm(sigma * lamb * v)
        b = np.dot(exp_k, exp_v)

        g[l][:,:] = np.dot(np.dot(b, g[l-1][:,:]), np.invert(b))
    return g



def save(model, beta, time_steps, gf):
    """ Save data to file

    To Do
    -----
    Improve saving and loading
    """
    file = f"data\\gf2_t={beta}_nt={time_steps}_{model.param_str()}"
    np.save(file, gf)


def measure(model, beta, time_steps):
    """ Runs the lqmc warmup and measurement loop for the given model.

    Parameters
    ----------
    model: HubbardModel
        The Hubbard model instance.
    beta: float
        The inverse temperature .math'\beta = 1/T'.
    time_steps: int
        Number of time steps from .math'0' to .math'\beta'

    Returns
    -------
    gf: (2, N) np.ndarray
        Measured Green's function .math'G' of the up- and down-spin channel.
    """
    dtau = beta / time_steps
    check_params(model.u, model.t, dtau)

    t0 = time.time()
    config = Configuration(model.n_sites, time_steps)
    config = warmup(model, config, dtau, sweeps=20)
    gf = measure_gf(model, config, dtau, sweeps=80)
    t = time.time() - t0

    mins, secs = divmod(t, 60)
    print(f"Total time: {int(mins):0>2}:{int(secs):0>2} min")
    print()
    save(model, beta, time_steps, gf)
    return gf


def filling(g_sigma):
    r""" Computes the local filling of the model.

    Parameters
    ----------
    g_sigma: (N) np.ndarray
        Green's function .math'G_{\sigma}' of a spin channel.

    Returns
    -------
    n: (N) np.ndarray
    """
    return 1 - np.diagonal(g_sigma)


def main():
    n_sites = 4
    u, t = 2, 1
    temp = 2
    beta = 1 / temp
    time_steps = 10

    model = HubbardModel(u=u, t=t, mu=u / 2)
    model.build(n_sites)

    gf_up, gf_dn = measure(model, beta, time_steps)
    # gf = np.load("data\\gf_t=2_nt=20_u=2_t=1_mu=1.0.npy")

    n_up = filling(gf_up)
    n_dn = filling(gf_dn)
    print(f"<n↑> = {np.mean(n_up):.3f}  {n_up}")
    print(f"<n↓> = {np.mean(n_dn):.3f}  {n_dn}")
    print(f"<n>  = {np.mean(n_up + n_dn):.3f}")


if __name__ == "__main__":
    main()

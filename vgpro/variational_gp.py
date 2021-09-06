from typing import Callable, Tuple, Union, Iterable, List

from jax import numpy as jnp, random, grad
from jax.lax import while_loop
from jax.ops import index_update
from jax.experimental.optimizers import OptimizerState, adam
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage


def gaussian_kernel(x1: Union[float, jnp.ndarray],
                    x2: Union[float, jnp.ndarray],
                    bandwidth: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Isotropic squared exponential kernel.

    Args:
        x1: Array of m points (m, d).
        x2: Array of n points (n, d).
        bandwidth: Kernel variance parameters (1,) or (d,).

    Returns:
        out: Array of kernel evaluations (m, n).
    """

    bandwidth = jnp.atleast_1d(bandwidth)
    x1 = jnp.atleast_2d(x1) / bandwidth
    x2 = jnp.atleast_2d(x2) / bandwidth
    sqdist = jnp.sum(x1 ** 2, -1).reshape(-1, 1) + jnp.sum(x2 ** 2, -1) - 2 * jnp.dot(x1, x2.T)
    kmat = jnp.exp(-0.5 * sqdist)
    return jnp.squeeze(kmat)


def logit(p: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    logit function maps [0,1] to R.

    Args:
        p: Array of n points in [0,1] (n,).

    Returns:
        out: Array of n points in R.
    """
    return jnp.log(p / (1 - p))


def invlogit(x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Inverse logit function maps R to [0,1].

    Args:
        x: Array of n points in R (n,).

    Returns:
        out: Array of n points in [0,1].
    """
    return 1 / (1 + jnp.exp(-x))


def logistic_log_likelihood(x: Union[float, jnp.ndarray],
                            y: Union[float, bool, jnp.ndarray],
                            theta: Union[float, bool, jnp.ndarray] = None,
                            extra_data: Union[float, bool, jnp.ndarray] = None) -> Union[float, jnp.ndarray]:
    """
    log likelihood for likelihood p(y|x) = Bernoulli(y | invlogit(x)).

    Args:
        x: Float or array in R.
        y: Float, bool or array of booleans or {0, 1}.
        theta: dummy variable for extra params.
        extra_data: dummy variable for extra data.

    Returns:
        out: log probability or probabilities of same length as x and y.
    """
    p = invlogit(x)
    p = jnp.where(y, p, 1 - p)
    return jnp.log(jnp.maximum(p, 1e-9))


def lower_triangular_to_array(tril_mat: jnp.ndarray) -> jnp.ndarray:
    """
    Flattens a lower triangular matrix into a 1D array.

    Args:
        tril_mat: Lower triangular matrix, (n, n).

    Returns:
        out: Flattened array (n(n+1)/2,).
    """
    return tril_mat[jnp.tril_indices_from(tril_mat)]


def array_to_lower_triangular(flat_arr: jnp.ndarray) -> jnp.ndarray:
    """
    Expands suitably sized array into lower triangular matrix:

    Args:
        flat_arr: Flat array of lower triangular values, (n(n+1)/2,).

    Returns:
        out: Lower triangular matrix, (n, n).
    """
    n = jnp.floor(jnp.sqrt(2 * flat_arr.size + 1)).astype('int32')
    tril_mat = jnp.eye(n)
    tril_mat = index_update(tril_mat, jnp.tril_indices_from(tril_mat), flat_arr)
    return tril_mat


def fit_variational_gp(x: jnp.ndarray,
                       y: jnp.ndarray,
                       kernel: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = gaussian_kernel,
                       initial_points: jnp.ndarray = None,
                       initial_mean: jnp.ndarray = None,
                       initial_sqrt_cov: jnp.ndarray = None,
                       initial_kernel_params: jnp.ndarray = None,
                       n_inducing_points: int = 32,
                       log_likelihood: Callable[
                           [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = logistic_log_likelihood,
                       initial_likelihood_params: jnp.ndarray = None,
                       extra_likelihood_data: jnp.ndarray = None,
                       gauss_hermite_degree: int = 20,
                       batch_size: int = None,
                       random_key: jnp.ndarray = None,
                       optimiser: Callable = adam,
                       step_size: float = 1e-3,
                       norm: int = jnp.inf,
                       gtol: float = 1e-5,
                       maxiter: int = None,
                       nugget: float = 1e-3,
                       **optim_params) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Tuple[int, float]]:
    """
    Fit variational sparse Gaussian Process.

    Args:
        x: Array of m points - input to GP (n, d_x).
        y: Array of n points - data for log_likelihood (n, d_y).
        kernel: Function that maps two arrays of points to a matrix of kernel evaluations.
        initial_points: Array of initial inducing point positions.
        initial_mean: Array of initial process values at initial inducing points.
        initial_sqrt_cov: Array for initial square root covariance at initial inducing points.
        initial_kernel_params: Array initial value for covariance kernel hyperparameters.
        n_inducing_points: Integer number of inducing/psuedo-points for variational distribution.
        log_likelihood: Function that takes Gaussian realisations in R, data from y
            and additional likelihood params + data then returns log probabilities of generating data.
        initial_likelihood_params: Array of initial extra likelihood params.
        extra_likelihood_data: Array of extra data for likelihood function (n, _).
        gauss_hermite_degree: Integer degree of Gauss-Hermite integration.
        batch_size: Integer minibatch size - default is n.
        random_key: Random key for subsampling - defaults is jax.random.PRNGKey(0).
        optimiser: Function from jax.experimental.optimizers.
        step_size: Float stepsize for optimiser.
        norm: Integer ord arg for jnp.linalg.norm to check convergence.
        gtol: Float tolerance for convergence of gradient norm.
        maxiter: Integer - max number of optimiser iterations - default 1000.
        nugget: Float (positive) to ensure Kernel matrix is invertible.

    Returns:
        points: Array of optimised inducing points (n_inducing_points, d_x)
        mean: Array mean of variational Gaussian at inducing points (n_inducing_points,)
        sqrt_cov: Lower triangular array square root of covariance as above (n_inducing_points, n_inducing_points)
        kernel_params: Array - optimised kernel parameters, same shape as initial_kernel_params
        theta: Array - additional optimised likelihood parameters, same shape as initial_likelihood_params
        (iter, final_norm_val) - tuple(number of iterations, final norm of gradient)
    """
    n, d = x.shape

    locs_mins = x.min(0)
    locs_maxs = x.max(0)
    locs_ranges = locs_maxs - locs_mins

    # Sample initial points uniformly on range of locs
    if initial_points is None:
        initial_points = locs_mins + locs_ranges * random.uniform(random.PRNGKey(0), shape=(n_inducing_points, d))
    else:
        n_inducing_points = len(initial_points)

    # Set boring standard mean and cov
    if initial_mean is None:
        initial_mean = jnp.zeros(n_inducing_points)
    if initial_sqrt_cov is None:
        initial_sqrt_cov = jnp.eye(n_inducing_points)

    if initial_kernel_params is None and kernel == gaussian_kernel:
        initial_kernel_params = jnp.ones(d)

    if initial_likelihood_params is None:
        initial_likelihood_params = jnp.array([])

    if extra_likelihood_data is None:
        extra_likelihood_data = jnp.empty(n)

    if random_key is None:
        random_key = random.PRNGKey(0)

    if batch_size is None:
        batch_size = n
        get_batch_inds = lambda _: jnp.arange(n)
    else:
        get_batch_inds = lambda rk: random.choice(rk, n, shape=(batch_size,))

    n_kernel_params = initial_kernel_params.size

    n_induc_lower_triangular = int(n_inducing_points * (n_inducing_points + 1) / 2)

    hg_x, hg_w = hermgauss(gauss_hermite_degree)
    hg_x = jnp.repeat(jnp.array(hg_x)[..., jnp.newaxis], batch_size, axis=1) * jnp.sqrt(2)  # shape (degree, batch_size)
    hg_w = jnp.array(hg_w)[..., jnp.newaxis] / jnp.sqrt(jnp.pi)  # shape (degree, 1)

    def concat(points: jnp.ndarray, mean: jnp.ndarray, sqrt_cov: jnp.ndarray,
               kern_params: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([jnp.concatenate(points),
                                mean,
                                lower_triangular_to_array(sqrt_cov),
                                kern_params,
                                theta])

    def array_to_sqrt_cov(flat_arr: jnp.ndarray) -> jnp.ndarray:
        tril_mat = jnp.eye(n_inducing_points)
        tril_mat = index_update(tril_mat, jnp.tril_indices_from(tril_mat), flat_arr)
        return tril_mat

    def expand(stack_arr: jnp.ndarray) -> Tuple:
        start_ind = 0
        points = stack_arr[start_ind:(start_ind + n_inducing_points * d)].reshape(n_inducing_points, d)
        start_ind += n_inducing_points * d
        mean = stack_arr[start_ind:(start_ind + n_inducing_points)]
        start_ind += n_inducing_points
        sqrt_cov = array_to_sqrt_cov(stack_arr[start_ind:(start_ind + n_induc_lower_triangular)])
        start_ind += n_induc_lower_triangular
        kernel_params = stack_arr[start_ind:(start_ind + n_kernel_params)]
        start_ind += n_kernel_params
        theta = stack_arr[start_ind:]
        return points, mean, sqrt_cov, kernel_params, theta

    def negative_elbo(points_mean_sqrt_cov_kern_theta_params: jnp.ndarray,
                      batch_inds: jnp.ndarray) -> float:
        points, mean, sqrt_cov, kern_params, theta = expand(points_mean_sqrt_cov_kern_theta_params)
        cov_m = sqrt_cov @ sqrt_cov.T

        knm = kernel(x[batch_inds], points, kern_params)

        kmm = kernel(points, points, kern_params) + nugget * jnp.eye(n_inducing_points)
        kmm_inv = jnp.linalg.inv(kmm)

        A = knm @ kmm_inv
        mean_n = A @ mean
        diag_cov_n = jnp.squeeze(kernel(jnp.zeros((1, 1)), jnp.zeros((1, 1)), kern_params)) \
                     + jnp.diag(A @ (cov_m - kmm) @ A.T)

        # Gauss-Hermite integration
        hg_n_x = jnp.sqrt(diag_cov_n) * hg_x + mean_n
        hg_log_lik_evals = log_likelihood(hg_n_x, y[batch_inds], theta, extra_likelihood_data[batch_inds])
        expected_log_lik_sum = (hg_w * hg_log_lik_evals).sum()

        kl_q_p = 0.5 * (jnp.trace(kmm_inv @ sqrt_cov @ sqrt_cov.T)
                        + mean.T @ kmm_inv @ mean
                        - n_inducing_points
                        + jnp.linalg.slogdet(kmm)[1] - jnp.log(jnp.prod(jnp.diag(sqrt_cov)) ** 2))

        return kl_q_p - expected_log_lik_sum

    grad_nelbo = grad(negative_elbo)

    init_carry = concat(initial_points, initial_mean, initial_sqrt_cov,
                        initial_kernel_params, initial_likelihood_params)

    if maxiter is None:
        maxiter = jnp.size(init_carry) * 200

    opt_init, opt_update, get_params = optimiser(step_size, **optim_params)
    opt_state = opt_init(init_carry)

    def body_fun(carry: Tuple[OptimizerState, int, float, jnp.ndarray]) \
            -> Tuple[OptimizerState, int, float, jnp.ndarray]:
        opt_state, ind, norm_val, rk = carry
        rk, subkey = random.split(rk)
        batch_inds = get_batch_inds(subkey)
        grads = grad_nelbo(get_params(opt_state), batch_inds)
        opt_state = opt_update(ind, grads, opt_state)
        return opt_state, ind + 1, jnp.linalg.norm(grads, ord=norm), rk

    def cond_fun(carry):
        return jnp.logical_and(carry[2] >= gtol,
                               carry[1] < maxiter)

    state = while_loop(cond_fun, body_fun, (opt_state, 0, jnp.inf, random_key))

    optim_points, optim_mean, optim_sqrt_cov, optim_kern_params, optim_theta = expand(get_params(state[0]))

    iters = state[1]
    final_norm_val = state[2]

    return optim_points, optim_mean, optim_sqrt_cov, optim_kern_params, optim_theta, (iters, final_norm_val)


def grid_to_points(grid: Iterable[jnp.ndarray]) -> jnp.ndarray:
    """
    Converts a grid object into a concatenated array.

    Args:
        grid: Length k iterable of size n arrays.

    Returns:
        out: (n, k) array
    """
    return jnp.array([jnp.concatenate(g) for g in grid]).T


def points_to_grid(points: jnp.ndarray, shape: tuple) -> List[jnp.ndarray]:
    """
    Converts single array into grid (list of arrays).

    Args:
        points: (n_x * n_y, k) array
        shape: Length k tuple describing shape of grid

    Returns:
        grid: Length k iterable of arrays each with shape=shape.

    """
    return [p.reshape(*shape) for p in points.T]


def plot_gp_heatmap(points: jnp.ndarray,
                    mean: jnp.ndarray,
                    sqrt_cov: jnp.ndarray,
                    kernel_params: jnp.ndarray,
                    kernel: Callable = gaussian_kernel,
                    transform: Callable = None,
                    x_linsp: jnp.ndarray = None,
                    y_linsp: jnp.ndarray = None,
                    resolution: int = 100,
                    nugget: float = 1e-3,
                    sd: bool = False,
                    ax: plt.Axes = None,
                    **kwargs) -> Tuple[plt.Axes, AxesImage]:
    """
    Plots 2D result from fit_variational_gp.

    Args:
        points: Inducing points, (m, 2) array.
        mean: Process mean at inducing points (m,) array.
        sqrt_cov: Lower triangular square root of covariance at inducing points (m, m) array.
        kernel_params: Kernel parameters (k,) array.
        kernel: Kernel function.
        transform: Function to transform the process values.
        x_linsp: Linspace corresponding to x-axis, defaults to jnp.linspace(x.min, x.max, resolution)
        y_linsp: Linspace corresponding to y-axis, defaults to jnp.linspace(y.min, y.max, resolution)
        resolution: Integer resolution for generating linsps.
        nugget: Float (positive) to ensure Kernel matrix is invertible.
        sd: bool whether to plot process standard deviations or means.
        ax: plt.Axes to plot on.
        **kwargs: extra kwargs for ax.imshow()

    Returns:
        ax: with heatmap added
        pos: color "mappable" object returned by ax.imshow

    """
    locs_mins = points.min(0)
    locs_maxs = points.max(0)

    n_points = len(points)

    if x_linsp is None:
        x_linsp = jnp.linspace(locs_mins[0], locs_maxs[0], resolution)
    x_len = len(x_linsp)

    if y_linsp is None:
        y_linsp = jnp.linspace(locs_mins[1], locs_maxs[1], resolution)
    y_len = len(y_linsp)

    grid = jnp.meshgrid(x_linsp, y_linsp)
    grid_points = grid_to_points(grid)

    knm = kernel(grid_points, points, kernel_params)
    kmm = kernel(points, points, kernel_params) + nugget * jnp.eye(n_points)
    kmm_inv = jnp.linalg.inv(kmm)
    A = knm @ kmm_inv

    if sd:
        grid_vals = jnp.sqrt(jnp.squeeze(kernel(jnp.zeros((1, 1)), jnp.zeros((1, 1)), kernel_params))
                             + jnp.diag(A @ (sqrt_cov @ sqrt_cov.T - kmm) @ A.T))
    else:
        grid_vals = A @ mean

    if transform is not None:
        grid_vals = transform(grid_vals, grid_points)

    if ax is None:
        _, ax = plt.subplots()

    pos = ax.imshow(grid_vals.reshape(x_len, y_len),
                    extent=[x_linsp.min(), x_linsp.max(), y_linsp.min(), y_linsp.max()], **kwargs)
    return ax, pos

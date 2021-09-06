from jax import numpy as jnp, random
import vgpro


def test_gaussian_kernel():
    assert vgpro.gaussian_kernel(0., 0., jnp.array([1., 1.])) == 1.
    assert jnp.all(vgpro.gaussian_kernel(jnp.zeros((10, 2)),
                                         jnp.zeros((20, 2)),
                                         jnp.array([1., 1.])) == jnp.ones((10, 20)))


def test_logit():
    assert vgpro.logit(0.5) == 0.
    assert jnp.all(vgpro.logit(0.5 * jnp.ones(10)) == jnp.zeros(10))


def test_invlogit():
    assert vgpro.invlogit(0.) == 0.5
    assert jnp.all(vgpro.invlogit(jnp.zeros(10)) == 0.5 * jnp.ones(10))


def test_logistic_log_likelihood():
    assert vgpro.logistic_log_likelihood(-jnp.inf, False) == jnp.log(1.)
    assert vgpro.logistic_log_likelihood(-jnp.inf, True) < -20

    assert vgpro.logistic_log_likelihood(0., False) == jnp.log(0.5)
    assert vgpro.logistic_log_likelihood(0., True) == jnp.log(0.5)

    assert vgpro.logistic_log_likelihood(jnp.inf, False) < -20
    assert vgpro.logistic_log_likelihood(jnp.inf, True) == jnp.log(1.)

    assert jnp.all(vgpro.logistic_log_likelihood(jnp.array([-jnp.inf, 0.]),
                                                 jnp.array([False, True])
                                                 == jnp.array([0., jnp.log(0.5)])))


def test_lower_triangular():
    d_id = 10
    id_mat = jnp.eye(d_id)
    flat_id_arr = vgpro.lower_triangular_to_array(id_mat)
    assert flat_id_arr.ndim == 1
    assert flat_id_arr.size == int(d_id * (d_id + 1) / 2)
    assert jnp.all(vgpro.array_to_lower_triangular(flat_id_arr) == id_mat)

    d_ones = 11
    ones_mat = jnp.tril(jnp.ones((d_ones, d_ones)))
    flat_ones_arr = vgpro.lower_triangular_to_array(ones_mat)
    assert flat_ones_arr.ndim == 1
    assert flat_ones_arr.size == int(d_ones * (d_ones + 1) / 2)
    assert jnp.all(vgpro.array_to_lower_triangular(flat_ones_arr) == ones_mat)


def test_fit_variational_gp_1d():
    n = 100
    d = 1
    x = random.normal(random.PRNGKey(0), shape=(n, d))
    y = x[:, 0] < 0

    fit = vgpro.fit_variational_gp(x, y, step_size=0.01, maxiter=100)
    points, mean, sqrt_cov, kernel_params, theta, (iter, final_grad_norm_val) = fit
    assert mean[points[:, 0] < 0].mean() > 0.
    assert mean[points[:, 0] > 0].mean() < 0.
    assert iter > 50
    assert final_grad_norm_val < 1000.


def test_fit_variational_gp_2d():
    n = 100
    d = 2
    x = random.normal(random.PRNGKey(0), shape=(n, d))
    y = x[:, 0] < 0

    fit = vgpro.fit_variational_gp(x, y, step_size=0.1, maxiter=100)
    points, mean, sqrt_cov, kernel_params, theta, (iter, final_grad_norm_val) = fit
    assert mean[points[:, 0] < 0].mean() > 0.
    assert mean[points[:, 0] > 0].mean() < 0.
    assert iter > 50
    assert final_grad_norm_val < 100.


def test_fit_variational_gp_2d_minibatch():
    n = 100
    d = 2
    x = random.normal(random.PRNGKey(0), shape=(n, d))
    y = x[:, 0] < 0

    fit = vgpro.fit_variational_gp(x, y, step_size=0.1, maxiter=100, batch_size=50)
    points, mean, sqrt_cov, kernel_params, theta, (iter, final_grad_norm_val) = fit
    assert mean[points[:, 0] < 0].mean() > 0.
    assert mean[points[:, 0] > 0].mean() < 0.
    assert iter > 50
    assert final_grad_norm_val < 100.

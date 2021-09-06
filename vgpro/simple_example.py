from jax import random
import matplotlib.pyplot as plt

import vgpro

n = 100
d = 2
x = random.normal(random.PRNGKey(0), shape=(n, d))
y = x[:, 0] < 0

maxiter = 500

fit = vgpro.fit_variational_gp(x, y, step_size=0.01, maxiter=maxiter)
points, mean, sqrt_cov, kernel_params, theta, (iter, final_grad_norm_val) = fit

x_cols = ['black' if a else 'red' for a in y]

# Plot probabilities
fig_m, ax_m = plt.subplots()
ax_m, pos_m = vgpro.plot_gp_heatmap(points, mean, sqrt_cov, kernel_params,
                                    transform=lambda a, _: vgpro.invlogit(a), ax=ax_m)
fig_m.colorbar(pos_m)
ax_m.scatter(x[:, 0], x[:, 1], color=x_cols)

# Plot uncertainty
fig_sd, ax_sd = plt.subplots()
ax_sd, pos_sd = vgpro.plot_gp_heatmap(points, mean, sqrt_cov, kernel_params, sd=True,
                                      transform=lambda a, _: vgpro.invlogit(a), ax=ax_sd, cmap='Purples')
fig_sd.colorbar(pos_sd)
ax_sd.scatter(x[:, 0], x[:, 1], color=x_cols)

plt.show()

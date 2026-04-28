"""
Two-cart system dynamics.

  F → [Cart 1] ===spring+damper=== [Cart 2] ← disturbance
       m1=1                         m2=1
  k=1 (spring), b=1 (damper), beta=1 (drag)
"""

import jax.numpy as jnp

# System parameters
m1 = 1.0
m2 = 1.0
k = 1.0
b = 1.0
beta = 1.0  # drag coefficient


def plant(q, dq, u, f_ext, m1=m1, m2=m2, k=k, b=b):
    """Two-cart plant dynamics.

    q = (x1, x2), dq = (dx1, dx2), u = F (scalar on cart 1),
    f_ext = (f1, f2) external forces on each cart.
    Returns ddq = (ddx1, ddx2).
    """
    spring_damper = k * (q[0] - q[1]) + b * (dq[0] - dq[1])
    ddx1 = (u - spring_damper + f_ext[0]) / m1
    ddx2 = (spring_damper + f_ext[1]) / m2
    return jnp.array([ddx1, ddx2])


def disturbance(q, dq, w, beta=beta):
    """Drag-like disturbance on both carts.

    w is the 'wind velocity' sampled from a distribution.
    Force on each cart: -beta * (dx_i - w) * |dx_i - w|  (quadratic drag)
    """
    dx1, dx2 = dq[0], dq[1]
    v_rel1 = dx1 - w
    v_rel2 = dx2 - w
    f_drag1 = -beta * v_rel1 * jnp.abs(v_rel1)
    f_drag2 = -beta * v_rel2 * jnp.abs(v_rel2)
    return jnp.array([f_drag1, f_drag2])

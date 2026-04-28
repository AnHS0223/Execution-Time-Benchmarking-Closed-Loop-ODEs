"""
Inverted Pendulum on Cart dynamics.

        θ  (angle from upright)
        |
        |  l=1
        o
  F → [Cart]
       M=1
  ──────────────────── ground

Control: horizontal force F on cart.
Track: pendulum angle θ near upright (θ=0).
"""

import jax.numpy as jnp

# System parameters
M = 1.0     # cart mass
m = 1.0     # pendulum mass
l = 1.0     # pendulum length
g = 9.81    # gravitational acceleration
beta = 0.1  # drag coefficient for wind disturbance


def plant(q, dq, u, f_ext, M=M, m=m, l=l, g=g):
    """Inverted pendulum on cart.

    q = (x, θ), dq = (dx, dθ), u = F (scalar, horizontal force on cart).
    f_ext = (f_cart, τ_pend) external forces.
    θ = 0 is upright.

    Equations of motion:
      (M+m)ẍ + mlθ̈cosθ - mlθ̇²sinθ = F + f_cart
      mlẍcosθ + ml²θ̈ - mglsinθ = τ_pend
    """
    x, θ = q[0], q[1]
    dx, dθ = dq[0], dq[1]
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)

    H = jnp.array([
        [M + m,      m*l*cosθ],
        [m*l*cosθ,   m*l**2  ],
    ])

    rhs = jnp.array([
        u + m*l*dθ**2*sinθ,
        m*g*l*sinθ + f_ext[1],
    ])

    ddq = jnp.linalg.solve(H, rhs)
    return ddq


def disturbance(q, dq, w, beta=beta, l=l):
    """Wind disturbance on pendulum tip.

    Wind velocity w creates drag on the pendulum tip,
    producing a torque about the pivot.
    """
    θ = q[1]
    dθ = dq[1]
    cosθ = jnp.cos(θ)
    # Horizontal velocity of tip relative to wind
    v_rel = l * dθ * cosθ - w
    # Quadratic drag → torque
    f_drag = -beta * v_rel * jnp.abs(v_rel)
    τ = f_drag * l * cosθ
    return jnp.array([0., τ])

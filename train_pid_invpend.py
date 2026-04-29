"""
PID Gain Tuning for Inverted Pendulum on Cart.

Control: force F on cart.  Tracking: pendulum angle θ near upright (θ=0).
Disturbance: wind torque on pendulum.
"""

from tqdm.auto import tqdm
import pickle
import time
from math import pi, inf
import os
import argparse
import functools
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
parser.add_argument('M', help='number of wind samples', type=int)
parser.add_argument('--use_x64', action='store_true')
args = parser.parse_args()

if args.use_x64:
    os.environ['JAX_ENABLE_X64'] = 'True'

import jax                                          # noqa: E402
# jax.config.update("jax_platforms", "cpu") # use CPU for benchmarking (optional)
import jax.numpy as jnp                             # noqa: E402
from jax.example_libraries import optimizers        # noqa: E402
from dynamics_invpend import plant, disturbance     # noqa: E402
from utils import (odeint_fixed_step,               # noqa: E402
                   random_ragged_spline, spline,
                   params_to_cholesky, params_to_posdef)

key = jax.random.PRNGKey(args.seed)

hparams = {
    'seed':             args.seed,
    'use_x64':          args.use_x64,
    'num_wind_samples': args.M,

    'pid': {
        'learning_rate':     1e-2,
        'num_steps':         500,
        'regularizer_ctrl':  1e-3,
        'train_frac':        0.75,
        'T':                 5.,
        'dt':                1e-2,
        'num_refs':          10,
        'num_knots':         6,
        'poly_orders':       (9,),          # 1 DOF (θ)
        'deriv_orders':      (4,),
        'min_step':          (-1.0,),       # cart position steps
        'max_step':          (1.0,),
        'min_ref':           (-5.0,),       # cart position bounds
        'max_ref':           (5.0,),
    },

    'wind': {
        'w_min': 0., 'w_max': 6., 'a': 5., 'b': 9.,
    },
}

if __name__ == "__main__":
    print('Inverted Pendulum PID Gain Tuning')
    print(f'  seed={args.seed}, M={args.M}, device={jax.devices()[0]}')

    num_track_dof = 1
    min_ref = jnp.asarray(hparams['pid']['min_ref'])
    max_ref = jnp.asarray(hparams['pid']['max_ref'])

    # ODE for PID closed-loop on inverted pendulum
    def ode(z, t, pid_params, w, reference, t_knots, coefs,
            plant=plant, disturbance=disturbance):
        x, ie, c = z
        q, dq = x[:2], x[2:]   # (x_cart, θ), (dx_cart, dθ)

        # Reference for cart x (scalar)
        r = reference(t)
        dr = jax.jacfwd(reference)(t)

        def ddr_fn(t):
            raw_ddr = jax.jacfwd(jax.jacfwd(reference))(t)
            dt_start = t_knots[1] - t_knots[0]
            init_ddr = (coefs[0][0, 2] * 2.0) / (dt_start**2)
            return jax.lax.cond(
                t <= 1e-10,
                lambda _: init_ddr,
                lambda _: raw_ddr,
                operand=None
            )
        ddr = ddr_fn(t)

        # Tracking errors: [x_err, θ_err]
        e = jnp.array([q[0] - r, q[1] - 0.])
        de = jnp.array([dq[0] - dr, dq[1] - 0.])

        # PID gains (raw vectors for underactuated system)
        Kp = pid_params['Kp']
        Kd = pid_params['Kd']
        Ki = pid_params['Ki']

        # PID control: force on cart
        u = -(jnp.dot(Kp, e) + jnp.dot(Kd, de) + jnp.dot(Ki, ie))

        # Real dynamics + wind disturbance
        f_ext = disturbance(q, dq, w)
        ddq = plant(q, dq, u, f_ext)
        dx = jnp.concatenate((dq, ddq))

        # Integral error derivative
        die = e

        # Cost terms (weight theta error higher to prioritize not falling)
        track_loss = e[0]**2 + de[0]**2 + 10.0 * (e[1]**2 + de[1]**2)
        dc = jnp.array([
            track_loss,   # tracking loss (x and θ)
            u**2,         # control loss
        ])

        dz = (dx, die, dc)
        return dz

    # Simulate across M wind samples for one reference
    def wind_sim(pid_params, w_samples, reference, t_knots, coefs, T, dt,
                 ode=ode):
        r0 = reference(0.)
        dr0 = jax.jacfwd(reference)(0.)
        # Cart at reference position, pendulum vertical
        x0 = jnp.array([r0, 0., dr0, 0.])  # x=r0, θ=0, dx=dr0, dθ=0
        ie0 = jnp.zeros(2)
        c0 = jnp.zeros(2)
        z0 = (x0, ie0, c0)

        ode_partial = jax.tree_util.Partial(
            ode, reference=reference, t_knots=t_knots, coefs=coefs)
        in_axes = (None, None, None, None, None, None, 0)
        z, t = jax.vmap(odeint_fixed_step, in_axes)(
            ode_partial, z0, 0., T, dt, pid_params, w_samples)
        x, ie, c = z
        return t, x, ie, c

    # Simulate across references × wind samples
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, None, None))
    def simulate(pid_params, w_samples, t_knots, coefs, T, dt,
                 min_ref=min_ref, max_ref=max_ref):
        def reference(t):
            r = spline(t, t_knots, coefs[0])
            r = jnp.clip(r, min_ref[0], max_ref[0])
            return r

        t, x, ie, c = wind_sim(pid_params, w_samples, reference,
                                t_knots, coefs, T, dt)
        return t, x, ie, c

    # Loss function
    @functools.partial(jax.jit, static_argnums=(4, 5))
    def loss(pid_params, w_samples, t_knots, coefs, T, dt,
             regularizer_ctrl):
        t, x, ie, c = simulate(pid_params, w_samples, t_knots, coefs, T, dt)
        c_final = jnp.sum(c[:, :, -1, :], axis=(0, 1))
        num_refs, num_winds = c.shape[0], c.shape[1]
        normalizer = T * num_refs * num_winds

        tracking_loss, control_loss = c_final
        total_loss = (tracking_loss
                      + regularizer_ctrl * control_loss) / normalizer
        aux = {
            'tracking_loss': tracking_loss / normalizer,
            'control_loss': control_loss / normalizer,
            'Kp': pid_params['Kp'],
            'Kd': pid_params['Kd'],
            'Ki': pid_params['Ki'],
        }
        return total_loss, aux

    # Generate 1D reference trajectories (for θ)
    num_refs = hparams['pid']['num_refs']
    key, *subkeys = jax.random.split(key, 1 + num_refs)
    subkeys = jnp.vstack(subkeys)
    in_axes = (0, None, None, None, None, None, None, None, None)
    t_knots, knots, coefs = jax.vmap(random_ragged_spline, in_axes)(
        subkeys,
        hparams['pid']['T'],
        hparams['pid']['num_knots'],
        hparams['pid']['poly_orders'],
        hparams['pid']['deriv_orders'],
        jnp.asarray(hparams['pid']['min_step']),
        jnp.asarray(hparams['pid']['max_step']),
        0.7 * min_ref,
        0.7 * max_ref,
    )

    # Sample M wind velocities
    M = hparams['num_wind_samples']
    wind_cfg = hparams['wind']
    key, subkey = jax.random.split(key, 2)
    w_samples = (wind_cfg['w_min']
                 + (wind_cfg['w_max'] - wind_cfg['w_min'])
                 * jax.random.beta(subkey, wind_cfg['a'], wind_cfg['b'],
                                   (M,)))

    # Train/valid split
    train_frac = hparams['pid']['train_frac']
    num_train_refs = int(train_frac * num_refs)
    num_train_winds = int(train_frac * M)

    train_t_knots = jax.tree_util.tree_map(lambda a: a[:num_train_refs], t_knots)
    train_coefs = jax.tree_util.tree_map(lambda a: a[:num_train_refs], coefs)
    valid_t_knots = jax.tree_util.tree_map(lambda a: a[num_train_refs:], t_knots)
    valid_coefs = jax.tree_util.tree_map(lambda a: a[num_train_refs:], coefs)
    train_w = w_samples[:num_train_winds]
    valid_w = w_samples[num_train_winds:]

    # Initialize PID gains (shape = 2 for [x, θ])
    pid_params = {
        'Kp': jnp.zeros(2),
        'Kd': jnp.zeros(2),
        'Ki': jnp.zeros(2),
    }

    # Adam optimizer
    learning_rate = hparams['pid']['learning_rate']
    init_opt, update_opt, get_params = optimizers.adam(learning_rate)
    opt_state = init_opt(pid_params)
    step_idx = 0
    best_idx = 0
    best_loss = jnp.inf
    valid_loss_history = []
    train_loss_history = []
    best_pid_params = pid_params

    @functools.partial(jax.jit, static_argnums=(5, 6))
    def step(idx, opt_state, w_samples, t_knots, coefs, T, dt,
             regularizer_ctrl):
        pid_params = get_params(opt_state)
        grads, aux = jax.grad(loss, argnums=0, has_aux=True)(
            pid_params, w_samples, t_knots, coefs, T, dt, regularizer_ctrl)
        opt_state = update_opt(idx, grads, opt_state)
        return opt_state, aux

    # Pre-compile
    print('Pre-compiling ... ', end='', flush=True)
    T = hparams['pid']['T']
    dt = hparams['pid']['dt']
    regularizer_ctrl = hparams['pid']['regularizer_ctrl']

    compile_start = time.time()
    _ = step(0, opt_state, train_w, train_t_knots, train_coefs,
             T, dt, regularizer_ctrl)
    _ = loss(pid_params, valid_w, valid_t_knots, valid_coefs, T, dt, 0.)
    compile_end = time.time()
    compile_time = compile_end - compile_start
    print('done ({:.2f} s)! Training ...'.format(compile_time))

    # Training loop
    start = time.time()
    for _ in tqdm(range(hparams['pid']['num_steps'])):
        opt_state, train_aux = step(
            step_idx, opt_state, train_w, train_t_knots, train_coefs,
            T, dt, regularizer_ctrl)
        new_pid_params = get_params(opt_state)

        valid_loss, valid_aux = loss(
            new_pid_params, valid_w, valid_t_knots, valid_coefs, T, dt, 0.)

        if valid_loss < best_loss:
            best_pid_params = new_pid_params
            best_loss = valid_loss
            best_idx = step_idx

        valid_loss_history.append(float(valid_loss))
        train_loss_history.append(float(train_aux['tracking_loss']))
        step_idx += 1

    end = time.time()
    training_time = end - start

    Kp_val = best_pid_params['Kp']
    Kd_val = best_pid_params['Kd']
    Ki_val = best_pid_params['Ki']

    print(f'done! ({training_time:.2f} s)')
    print(f'  Compile: {compile_time:.2f}s, Train: {training_time:.2f}s')
    print(f'  Best step: {best_idx}, Best valid loss: {best_loss:.8f}')
    print(f'  Kp={Kp_val}, Kd={Kd_val}, Ki={Ki_val}')

    # Save
    output_name = "invpend_pid_seed={:d}_M={:d}".format(args.seed, M)
    results = {
        'best_step_idx': best_idx,
        'valid_loss_history': valid_loss_history,
        'train_loss_history': train_loss_history,
        'hparams': hparams,
        'controller': {
            'Kp': best_pid_params['Kp'],
            'Kd': best_pid_params['Kd'],
            'Ki': best_pid_params['Ki'],
        },
        'training_time': training_time,
        'compile_time': compile_time,
        'device': str(jax.devices()[0]),
    }
    os.makedirs('train_results', exist_ok=True)
    output_path = os.path.join('train_results', output_name + '.pkl')
    with open(output_path, 'wb') as file:
        pickle.dump(results, file)
    print(f'Saved to {output_path}')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss_history, label='Train', alpha=0.8)
    ax.plot(valid_loss_history, label='Valid', alpha=0.8)
    ax.axvline(x=best_idx, color='r', linestyle='--', alpha=0.5,
               label=f'Best={best_idx}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tracking Loss')
    ax.set_title(f'InvPend PID (seed={args.seed}, M={M})')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plot_path = os.path.join('train_results', output_name + '_loss.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Plot saved to {plot_path}')

from .lifting import WrapModel, create_library, krtimes
from .optimization import quad_program

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import line_search
from scipy.sparse import block_diag
from tqdm.auto import tqdm
import warnings

import matplotlib.pyplot as plt
import os


class StepClock:
    def __init__(self, dt, horizon, n_steps):
        self.dt = float(dt)
        self.horizon = horizon
        self.n_steps = n_steps
        self.ts = np.linspace(0, self.dt * self.n_steps, self.n_steps, endpoint=False)
        self.ts_sim = self.ts

    def set_endsim(self, index):
        self.ts_sim = self.ts[:index]

    def ts_step(self, a_step):
        return np.linspace(self.dt * a_step, self.dt * (a_step + 1), 2)

    def ts_horizon(self, a_step):
        return np.linspace(self.dt * a_step, self.dt * (a_step + self.horizon), self.horizon, endpoint=False)

    def to_string(self):
        labels = ['dt', val_to_str(self.dt)] + ['h', val_to_str(self.horizon)] + ['n', val_to_str(self.n_steps)]
        return '_'.join(labels)


def val_to_str(val):
    str_val = f'{val:.1E}'
    str_val = str_val.replace('E', 'e').replace('.','d')
    str_val = str_val.replace('-','m').replace('+','')
    return str_val


def shift_guess(data):
    _, n = data.shape
    return np.hstack([data[:, 1:].reshape(-1, n - 1), data[:, -1].reshape(-1, 1)])


def real_to_complex(z):
    # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]


def complex_to_real(z):
    # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def complex_to_real_op(Z):
    return np.block([[Z.real, -Z.imag], [Z.imag, Z.real]])


def real_to_complex_op(Z):
    row, col = Z.shape
    return Z[:row//2, :col//2] + 1j * Z[row//2:, :col//2]


def mpc(x0, dim_u, order, X_targ, U_targ, clock, experiment, model, Q, R, Qf, sat=None, du=None, max_iter=100,
        exit_condition=None, streaming=False, progress_bar=True, verbose=False):
    # Set default mpc exit
    exit_code = 0

    # Initialize
    # ==========
    xs = [None] * (clock.n_steps + 1)
    us = [None] * clock.n_steps

    # Set guess to initial value (a la SDRE)
    # Note: Could also use targets. This would lead to traditional MPC.
    X_guess = np.hstack([x0.reshape(-1, 1)] * (clock.horizon + 1))
    U_guess = np.hstack([np.zeros([dim_u, 1])] * clock.horizon)

    # Set initial benchmarks
    X_htarg = np.atleast_2d(X_targ[:, :clock.horizon + 1])
    U_htarg = np.atleast_2d(U_targ[:, :clock.horizon])

    # Stretch Q, R
    Q_ls = [Q] * clock.horizon
    Q_ls.append(Qf)
    R_ls = [R] * clock.horizon

    # Wrap the discretized model to allow for local approximations later
    # Note 1: We discretize the continuous dynamics, then linearize around the guess. Could invert this.
    # Note 2: This idea could be improved via automatic differentiation of more mature simulations.
    wrapped_model = WrapModel(*model.get_discrete(), dim_u, order)

    # Solve MPC
    # =========
    xs[0] = x0
    for step in tqdm(range(clock.n_steps)) if progress_bar else range(clock.n_steps):
        # Iterative QP
        # ------------
        n_iter = 0
        iqp_exit_condition = False

        # DIAGNOSTIC
        _save_control = []
        _save_state = []
        _gradient_list = []
        _obj_list = []

        while not iqp_exit_condition and n_iter < max_iter:
            A_ls, B_ls, Delta_ls = wrapped_model.get_model_along_traj(X_guess, U_guess, clock.ts_horizon(step))

            # Run QP
            # ^^^^^^
            with warnings.catch_warnings():
                # Catch deprecation of np.complex in cvxpy
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                # Catch bad optimization warning in cvxpy
                warnings.simplefilter(action="error", category=UserWarning)
                try:
                    u_prev = us[step - 1] if step > 1 else U_htarg[:, 0].reshape(-1, 1)
                    # N.b. solving for x, u instead of dx, du.
                    X_opt, U_opt, obj_val, prob = quad_program(xs[step], X_htarg, U_htarg,
                                                               Q_ls, R_ls,
                                                               A_ls, B_ls, Delta_ls,
                                                               u_prev, sat, du, verbose)
                except Warning as w:
                    print(w)
                    exit_code = 2
                    break

            # Warn if failed convergence
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^
            if np.isinf(obj_val):
                warnings.warn("Solution was infinite (failed to converge). Inspect the model for accuracy, "
                              "check if control constraints can regularize the problem, "
                              "or run with verbose=True for more information.")
                exit_code = 3
                break

            # Line search
            # ^^^^^^^^^^^
            # Line search looks for an optimal step length alpha in the direction *_opt - *_guess
            if step > 0:
                # Assume that the shifted solutions are not far from the optimum. Take the full step.
                alpha = 1
                iqp_exit_condition = True
            else:
                # Constants (convert imaginary to real)
                Big_Cost = block_diag([complex_to_real_op(iq) for iq in Q_ls] +
                                      [complex_to_real_op(ir) for ir in R_ls]).tocsr()
                Z_htarg = np.concatenate((complex_to_real(X_htarg.flatten()), complex_to_real(U_htarg.flatten())))
                Z_guess = np.concatenate((complex_to_real(X_guess.flatten()), complex_to_real(U_guess.flatten())))
                Z_opt = np.concatenate((complex_to_real(X_opt.flatten()), complex_to_real(U_opt.flatten())))

                def fn(Z):
                    return (Z - Z_htarg) @ Big_Cost.dot(Z - Z_htarg) / 2

                def grad_fn(Z):
                    return (Big_Cost + Big_Cost.T).dot(Z - Z_htarg) / 2

                def hess_fn(Delta_Z):
                    return (Big_Cost + Big_Cost.T)/2

                # Direct line search: d f(z + alpha * dz) / d alpha != 0
                DZ = Z_opt - Z_guess
                alpha = - grad_fn(Z_guess).dot(DZ) / (DZ @ hess_fn(Z_guess).dot(DZ))
   
                # # DIAGNOSTIC
                # new_fval = fn(Z_guess + alpha * DZ)
                # new_slope = grad_fn(Z_guess + alpha * DZ)
                # _gradient_list.append([np.linalg.norm(new_slope), np.linalg.norm(DZ)])
                # _obj_list.append([new_fval, alpha])

                # Check convergence (absolute tolerance)
                if alpha * np.linalg.norm(DZ) < 1e-4:
                    iqp_exit_condition = True

            # Update guess along step.
            X_guess = X_guess + alpha * (X_opt - X_guess)
            U_guess = U_guess + alpha * (U_opt - U_guess)
            n_iter += 1

            # DIAGNOSTIC
            _save_control.append(U_guess)
            _save_state.append(X_guess)

        # Status check
        # ------------
        # quad_program failure?
        if exit_code > 0:
            break

        # # DIAGNOSTIC
        # if n_iter > 1:
        #     path = '../playground/Plot_NMPC/clock_{}/'.format(clock.to_string())
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #     fig, axes = plt.subplots(2, 1)
        #     fig.suptitle('order={}, step={}, iter={}'.format(order, step, n_iter))
        #     for i, control in enumerate(_save_control):
        #         ax = axes[0]
        #         ax.step(np.arange(len(control[0]) + 1), np.hstack([control[0], control[0][-1]]), color='k',
        #                 alpha=(i + 1) / (len(_save_control)), where='post')
        #     ax = axes[1]
        #     # np.arange(len(_save_state)), [np.linalg.norm(s - X_htarg, 2) for s in _save_state]
        #     _obj_list = np.vstack(_obj_list).T
        #     ax.plot(_obj_list[0] * _obj_list[1])
        #     ax.set_yscale('log')
        #     fig.savefig(path + 'seq_order{}_step{}_iter{}.png'.format(order, step, n_iter))

        # Simulate
        # --------
        # TODO: Is xs a list of model states or experiment states? Equiv., what is experiment class's input/output?
        # -- Apply the control to the experiment (lift/proj to convert between model state and simulation state).
        # -- I.e., next_xk = experiment.lift(experiment.simulate(xk, tk, uk))
        # -- Alternatively, close the loop with the model: next_xk = model.predict(xk, krtimes(lift(uk), xk))
        us[step] = U_opt[:, 0]
        ts_step = clock.ts_step(step)
        u_fns = interp1d(ts_step, np.vstack([us[step], us[step]]).T, fill_value='extrapolate', kind='previous',
                         assume_sorted=True)
        result = experiment.simulate(xs[step], ts_step, u_fns)
        xs[step + 1] = result[:, -1]

        # Shift guess
        # -----------
        X_guess = shift_guess(X_guess)
        U_guess = shift_guess(U_guess)

        # Shift targets
        # -------------
        X_htarg = np.atleast_2d(X_targ[:, step:step + clock.horizon + 1])
        U_htarg = np.atleast_2d(U_targ[:, step:step + clock.horizon])

        # Online model update
        # -------------------
        if streaming:
            fns = create_library(order, dim_u)[1:]
            lift_u = np.vstack([f(us[step].reshape(-1, 1)) for f in fns])
            model.fit_iteration(xs[step + 1], xs[step], krtimes(lift_u, xs[step].reshape(-1, 1)))

        # Finish
        # ------
        if exit_condition is not None:
            if exit_condition(xs[step + 1], xs[step], us[step]):
                exit_code = 1
                break

    if exit_code == 0:
        # Normal exit
        clock.set_endsim(step + 1)
        return [np.vstack(xs[:step + 2]).T, np.vstack(us[:step + 1]).T], model, exit_code
    else:
        # Early exit (ignore last attempted entry)
        clock.set_endsim(step)
        if step == 0:
            return [np.vstack(xs[:step + 1]).T, None], model, exit_code
        else:
            return [np.vstack(xs[:step + 1]).T, np.vstack(us[:step]).T], model, exit_code

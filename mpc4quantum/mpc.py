from .lifting import WrapModel, create_library, krtimes

import cvxpy as cp
import numpy as np
from scipy.sparse import block_diag
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
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


def quad_program(x0, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, u_prev=None, sat=None, verbose=False):
    # Shapes
    dim_u, n_steps = U_bm.shape
    dim_x, _ = X_bm.shape

    # Variables
    X = cp.Variable((dim_x, n_steps + 1), complex=True)
    U = cp.Variable([dim_u, n_steps])

    # Init QP
    cost = 0
    constr = []
    constr += [X[:, 0] == x0]

    # QP
    for t in range(n_steps):
        cost += cp.quad_form(X[:, t] - X_bm[:, t], Q_ls[t]) + cp.quad_form(U[:, t] - U_bm[:, t], R_ls[t])
        constr += [X[:, t + 1] == A_ls[t] @ X[:, t] + B_ls[t] @ U[:, t]]
        constr += [cp.norm(U[:, t], 'inf') <= sat]
    # Catch small bug in cvxpy related to zero quad form
    if np.any(Q_ls[-1] > 0):
        cost += cp.quad_form(X[:, -1] - X_bm[:, -1], Q_ls[-1])

    # Pin control to u_prev (regularize gradients)
    if u_prev is not None:
        constr += [cp.norm(U[:, 0] - u_prev.flatten(), 1) <= 0.5]

    prob = cp.Problem(cp.Minimize(cp.real(cost)), constr)
    obj_val = prob.solve(solver=cp.ECOS, verbose=verbose)
    return X.value, U.value, obj_val, prob

# def quad_program(x0, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, u_prev=None, sat=None, verbose=False):
#     """
#     Note the following error:
#         ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has
#         1 dimension(s) and the array at index 1 has 2 dimension(s)
#
#     If you see this error, then cvxpy is unable to handle the fact that Q_ls[i] may be zero. This error appears when
#     converting the quadratic form to a norm which is what conic solvers do.
#     """
#     # Shapes
#     dim_u, n_steps = U_bm.shape
#     dim_x, _ = X_bm.shape
#
#     # Flat variables
#     X = cp.Variable((dim_x * (n_steps + 1)), complex=True)
#     U = cp.Variable((dim_u * n_steps))
#
#     # Cost
#     cost = 0
#     R = block_diag(R_ls)
#     cost += cp.quad_form(U - U_bm.T.flatten(), R)
#     # note: conic solvers use norms not quadratic forms (manually replace <x,Qx> bc Q >= 0 can have bad numerics)
#     sqrt_Q = block_diag([sqrtm(q) for q in Q_ls])
#     cost += cp.norm(sqrt_Q @ (X - X_bm.T.flatten()), 2)**2
#
#     # Dynamics constraints
#     # note: individual constraints for each time improve optimization success (empirical claim);
#     #       this is in contrast to a single RHS == LHS for all times.
#     constr = []
#     constr += [X[:dim_x] == x0]
#     for t in range(n_steps):
#         RHS = A_ls[t] @ X[t * dim_x: (t + 1) * dim_x] + B_ls[t] @ U[t * dim_u: (t + 1) * dim_u]
#         LHS = X[(t + 1) * dim_x: (t + 2) * dim_x]
#         constr += [LHS == RHS]
#         # constr += [cp.norm(LHS - RHS, 'inf') <= 1e-4]
#
#     # Control constraint
#     # TODO: Better design!
#     if sat is not None:
#         constr += [cp.norm(U, 'inf') <= sat]
#     # if u_prev is not None:
#         # constr += [cp.norm(U[:dim_u] - u_prev.flatten(), 1) <= 0.5]
#
#     prob = cp.Problem(cp.Minimize(cp.real(cost)), constr)
#     obj_val = prob.solve(solver=cp.ECOS, verbose=verbose)
#     X_reshape = X.value if X.value is None else X.value.reshape(n_steps + 1, dim_x).T
#     U_reshape = U.value if U.value is None else U.value.reshape(n_steps, dim_u).T
#     return X_reshape, U_reshape, obj_val, prob


def shift_guess(data):
    _, n = data.shape
    return np.hstack([data[:, 1:].reshape(-1, n - 1), data[:, -1].reshape(-1, 1)])


def mpc(x0, dim_u, order, X_bm, U_bm, clock, experiment, model, Q, R, Qf, sat=None, max_iter=10, exit_condition=None,
        streaming=False, progress_bar=True, verbose=False):
    # Set default mpc exit
    exit_code = 0

    # Initialize
    # ==========
    xs = [None] * (clock.n_steps + 1)
    us = [None] * clock.n_steps

    # Set guess to initial value (a la SDRE)
    X_guess = np.hstack([x0.reshape(-1, 1)] * (clock.n_steps + 1))
    U_guess = np.hstack([np.zeros([dim_u, 1])] * clock.n_steps)

    # Stretch Q, R
    Q_ls = [Q] * clock.n_steps
    Q_ls.append(Qf)
    R_ls = [R] * clock.n_steps

    # Wrap (A, N) model to permit local approximations
    # TODO: We want the ability to also use linear models in this way.
    wrapped_model = WrapModel(*model.get_discrete(), dim_u, order)

    # Solve MPC
    # =========
    xs[0] = x0
    for a_step in tqdm(range(clock.n_steps)) if progress_bar else range(clock.n_steps):
        # Iterative QP
        # ------------
        n_iter = 0
        iqp_exit_condition = False
        obj_prev = np.infty
        # DIAGNOSTIC
        # _save_control = []
        # _save_state = []
        while not iqp_exit_condition and n_iter < max_iter:
            A_ls, B_ls = wrapped_model.get_model_along_traj(X_guess, U_guess, clock.ts)

            # Run QP
            # ^^^^^^
            with warnings.catch_warnings():
                # Catch deprecation of np.complex
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                # Catch bad optimization warning
                warnings.simplefilter(action="error", category=UserWarning)
                try:
                    u_prev = us[a_step - 1] if a_step > 1 else np.zeros((dim_u, 1))
                    X_opt, U_opt, obj_val, prob = quad_program(xs[a_step], X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls,
                                                               u_prev, sat, verbose)
                except Warning as w:
                    print(w)
                    exit_code = 2
                    break
            # Catch failure
            # ^^^^^^^^^^^^^
            if obj_val is np.inf:
                warnings.warn("Solution was infinite (failed to converge). Inspect the model for accuracy, "
                              "check if control constraints can regularize the problem, "
                              "or run with verbose=True for more information.")
                exit_code = 3
                break

            # Check convergence
            # ^^^^^^^^^^^^^^^^^
            if obj_prev < obj_val or np.isclose(obj_prev, obj_val, rtol=1e-05, atol=1e-08):
                iqp_exit_condition = True
            else:
                # Update
                X_guess = X_opt
                U_guess = U_opt
                # DIAGNOSTIC
                # _save_control.append(U_opt)
                # _save_state.append(X_guess)
                obj_prev = obj_val
                n_iter += 1

        # Status check
        # ------------
        # Check for a quad_program failure
        if exit_code > 0:
            break

        # DIAGNOSTIC
        # if n_iter > 1:
        #     path = '../playground/sequential_dt{}{}/'.format(*str(clock.dt).split('.'))
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #     fig, axes = plt.subplots(2, 1)
        #     fig.suptitle('order={}, a_step={}, iter={}'.format(order, a_step, n_iter))
        #     for i, control in enumerate(_save_control):
        #         ax = axes[0]
        #         ax.step(np.arange(len(control[0])), control[0], color='k', alpha=(i + 1) / (len(_save_control)))
        #     ax = axes[1]
        #     ax.plot(np.arange(len(_save_state)), [np.linalg.norm(s - X_bm, 2) for s in _save_state])
        #     fig.savefig(path + 'seq_order{}_step{}_iter{}.png'.format(order, a_step, n_iter))

        # Simulate
        # --------
        # -- Apply the control to the experiment (lift/proj to convert between model state and simulation state).
        # -- Alternatively, close the loop with the model: model.predict(xk, krtimes(lift(uk), xk))
        us[a_step] = U_opt[:, 0]
        ts_step = clock.ts_step(a_step)
        u_fns = interp1d(ts_step, np.vstack([us[a_step], us[a_step]]).T, fill_value='extrapolate', kind='previous',
                         assume_sorted=True)
        result = experiment.simulate(experiment.proj(xs[a_step]), ts_step, u_fns)
        xs[a_step + 1] = result[:, -1]

        # Shift guess
        # -----------
        X_guess = shift_guess(X_guess)
        U_guess = shift_guess(U_guess)

        # Online model update
        # -------------------
        if streaming:
            fns = create_library(order, dim_u)[1:]
            lift_u = np.vstack([f(us[a_step].reshape(-1, 1)) for f in fns])
            model.fit_iteration(xs[a_step + 1], xs[a_step], krtimes(lift_u, xs[a_step].reshape(-1, 1)))

        # Finish
        # ------
        if exit_condition is not None:
            if exit_condition(xs[a_step + 1], xs[a_step], us[a_step]):
                exit_code = 1
                break

    if exit_code == 0:
        # Normal exit
        clock.set_endsim(a_step + 1)
        return [np.vstack(xs[:a_step + 2]).T, np.vstack(us[:a_step + 1]).T], model, exit_code
    else:
        # Early exit (ignore last attempted entry)
        clock.set_endsim(a_step)
        if a_step == 0:
            return [np.vstack(xs[:a_step + 1]).T, None], model, exit_code
        else:
            return [np.vstack(xs[:a_step + 1]).T, np.vstack(us[:a_step]).T], model, exit_code

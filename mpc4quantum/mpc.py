from .lifting import WrapModel, create_library, krtimes

import cvxpy as cp
import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import warnings


class StepClock:
    def __init__(self, dt, horizon, n_steps):
        self.dt = dt
        self.horizon = horizon
        self.n_steps = n_steps
        self.ts = np.linspace(0, self.dt * self.n_steps, self.n_steps, endpoint=False)
        self.ts_sim = self.ts

    def set_endsim(self, index):
        self.ts_sim = self.ts[:index]

    def ts_step(self, a_step):
        return np.linspace(self.dt * a_step, self.dt * (a_step + 1), 2)


# def _quad_form(x, P):
#     # Check for zero matrix before compute
#     return cp.quad_form(x, P) if np.any(P > 0) else []


def quad_program(x0, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, verbose=False):
    """
    Solve a quadratic program under X' = A X + B U.

    :param x0: Initial (flat)
    :param X_bm: dim_x by n_steps + 1
    :param U_bm: dim_u by n_steps
    :param Q_ls: List of state costs. Length n_steps + 1.
    :param R_ls: List of control costs. Length n_steps.
    :param A_ls: List of drift operators. Length n_steps.
    :param B_ls: List of control operators. Length n_steps.
    :param verbose: Print cvxpy diagnostics. Default False.
    :return: Solved quadratic program for the dynamics.
    """
    # Shapes
    dim_u, n_steps = U_bm.shape
    dim_x, _ = X_bm.shape

    # Variables
    X = cp.Variable((dim_x, n_steps + 1), complex=True)
    U = cp.Variable([dim_u, n_steps])

    # QP
    cost = 0
    constr = []
    for t in range(n_steps):
        cost += cp.quad_form(X[:, t] - X_bm[:, t], Q_ls[t]) + cp.quad_form(U[:, t] - U_bm[:, t], R_ls[t])
        constr += [X[:, t + 1] == A_ls[t] @ X[:, t] + B_ls[t] @ U[:, t]]
        # TODO: manual control constraints should be coded differently.
        constr += [cp.norm(U[:, t]) <= 10]
        if t > 1:
            constr += [cp.norm(U[:, t] - U[:, t-1]) <= 5]
    constr += [X[:, 0] == x0]
    # Catch small bug in cvxpy related to zero quad form
    if np.any(Q_ls[-1] > 0):
        cost += cp.quad_form(X[:, -1] - X_bm[:, -1], Q_ls[-1])
    prob = cp.Problem(cp.Minimize(cp.real(cost)), constr)
    obj_val = prob.solve(solver=cp.ECOS, verbose=verbose)
    return X.value, U.value, obj_val, prob


def shift_guess(data):
    _, n = data.shape
    return np.hstack([data[:, 1:].reshape(-1, n - 1), data[:, -1].reshape(-1, 1)])


def mpc(x0, dim_u, order, X_bm, U_bm, clock, experiment, model, Q, R, Qf, max_iter=10, streaming=False,
        progress_bar=True, verbose=False):
    # Set an mpc exit to stop early
    mpc_exit = False
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
    for k in tqdm(range(clock.n_steps)) if progress_bar else range(clock.n_steps):
        # Iterative QP
        # ------------
        n_iter = 0
        iqp_exit_condition = False
        while not iqp_exit_condition and n_iter < max_iter:
            A_ls, B_ls = wrapped_model.get_model_along_traj(X_guess, U_guess, clock.ts)
            with warnings.catch_warnings():
                # Catch deprecation of np.complex
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                # Catch bad optimization warning
                warnings.simplefilter(action="error", category=UserWarning)
                try:
                    X_opt, U_opt, obj_val, prob = quad_program(xs[k], X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, verbose)
                except Warning as w:
                    print(w)
                    exit_code = 2
                    break
            # Check outcome for failed convergence
            if obj_val is np.inf:
                warnings.warn("Quadratic program failed to converge. Inspect the model.")
                exit_code = 3
                break

            # Exit condition for iteration convergence
            iqp_exit_condition = k > 0
            n_iter += 1

        # TODO: Is there anything else we should do for truncated runs?
        # Check for quad_program success / failure
        if exit_code > 0:
            break

        # Simulate
        # --------
        # -- Apply the control to the experiment (lift/proj to convert between model state and simulation state).
        # -- Alternatively, close the loop with the model: model.predict(xk, krtimes(lift(uk), xk))
        us[k] = U_opt[:, 0]
        ts_k = clock.ts_step(k)
        u_fns = interp1d(ts_k, np.vstack([us[k], us[k]]).T, fill_value='extrapolate', kind='previous',
                         assume_sorted=True)
        result = experiment.simulate(experiment.proj(xs[k]), ts_k, u_fns)
        xs[k + 1] = result[:, -1]

        # Shift guess
        # -----------
        X_guess = shift_guess(X_opt)
        U_guess = shift_guess(U_opt)

        # Online model update
        # -------------------
        if streaming:
            fns = create_library(order, dim_u)[1:]
            lift_u = np.vstack([f(us[k].reshape(-1, 1)) for f in fns])
            model.fit_iteration(xs[k + 1], xs[k], krtimes(lift_u, xs[k].reshape(-1, 1)))

        if mpc_exit:
            exit_code = 1
            break

    clock.set_endsim(k)
    return [np.vstack(xs[:k + 1]).T, np.vstack(us[:k]).T], model, exit_code


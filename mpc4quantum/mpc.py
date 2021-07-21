from .lifting import WrapModel, create_library, krtimes

import numpy as np
import warnings
from scipy.interpolate import interp1d
import cvxpy as cp


class StepClock:
    def __init__(self, dt, horizon, n_steps):
        self.dt = dt
        self.horizon = horizon
        self.n_steps = n_steps
        self.ts = np.linspace(0, self.dt * self.n_steps, self.n_steps, endpoint=False)

    def ts_step(self, a_step):
        return np.linspace(self.dt * a_step, self.dt * (a_step + 1), 2)


def quad_program(x0, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls):
    """
    Solve a quadratic program under X' = A X + B U.

    :param x0: Initial (flat)
    :param X_bm: x_dim by n_steps + 1
    :param U_bm: u_dim by n_steps
    :param Q_ls: List of state costs. Length n_steps + 1.
    :param R_ls: List of control costs. Length n_steps.
    :param A_ls: List of drift operators. Length n_steps.
    :param B_ls: List of control operators. Length n_steps.
    :return: Solved quadratic program for the dynamics.
    """
    # Shapes
    u_dim, n_steps = U_bm.shape
    x_dim, _ = X_bm.shape

    # Variables
    X = cp.Variable((x_dim, n_steps + 1), complex=True)
    U = cp.Variable([u_dim, n_steps])

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
    cost += cp.quad_form(X[:, -1] - X_bm[:, -1], Q_ls[-1])
    prob = cp.Problem(cp.Minimize(cp.real(cost)), constr)
    val = prob.solve(solver=cp.ECOS)
    return X.value, U.value, [val, prob]


def shift_guess(data):
    _, n = data.shape
    return np.hstack([data[:, 1:].reshape(-1, n - 1), data[:, -1].reshape(-1, 1)])


def mpc(x0, u_dim, order, X_bm, U_bm, clock, experiment, model, Q, R, Qf, max_iter=10, streaming=False):
    # Set an mpc exit to stop early
    mpc_exit = False
    exit_code = 0

    # Initialize
    # ==========
    xs = [None] * (clock.n_steps + 1)
    us = [None] * clock.n_steps

    # Set guess to initial value (a la SDRE)
    X_guess = np.hstack([x0.reshape(-1, 1)] * (clock.n_steps + 1))
    U_guess = np.hstack([np.zeros([u_dim, 1])] * clock.n_steps)

    # Stretch Q, R
    Q_ls = [Q] * clock.n_steps
    Q_ls.append(Qf)
    R_ls = [R] * clock.n_steps

    # Wrap (A, N) model to permit local approximations
    # TODO: We want the ability to also use linear models in this way.
    wrapped_model = WrapModel(*model.get_discrete(), u_dim, order)

    # Solve MPC
    # =========
    xs[0] = x0
    for k in range(clock.n_steps):
        # Iterative QP
        # ------------
        n_iter = 0
        iqp_exit = False
        while not iqp_exit and n_iter < max_iter:
            A_ls, B_ls = wrapped_model.get_model_along_traj(X_guess, U_guess, clock.ts)
            # Catch deprecation of np.complex
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                X_opt, U_opt, _ = quad_program(xs[k], X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls)
            # iqp_exit = False
            iqp_exit = k > 0 # -- tmp. soln: Only fit first pass.
            n_iter += 1

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
            fns = create_library(order, u_dim)[1:]
            lift_u = np.vstack([f(us[k].reshape(-1, 1)) for f in fns])
            model.fit_iteration(xs[k + 1], xs[k], krtimes(lift_u, xs[k].reshape(-1, 1)))

        if mpc_exit:
            exit_code = k
            break

    return [np.vstack(xs[:k + 1]).T, np.vstack(us[:k + 1]).T], model, exit_code


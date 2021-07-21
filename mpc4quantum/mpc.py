import numpy as np
import cvxpy as cp


class KeepTime:
    def __init__(self, dt, horizon, n_steps):
        self.dt = dt
        self.horizon = horizon
        self.n_steps = n_steps
        self.ts_sim = np.linspace(0, self.dt * self.n_steps, self.n_steps + 1)

    def ts_step(self, i_step):
        return np.linspace(self.dt * i_step, self.dt * (i_step + 1), 2)


def quad_program(x0, Xb, Ub, Q_ls, R_ls, A_ls, B_ls):
    """
    Solve a quadratic program.

    :param x0: Initial (flat)
    :param Xb: x_dim by n_steps + 1
    :param Ub: u_dim by n_steps
    :param Q_ls: List of state costs. Length n_steps + 1.
    :param R_ls: List of control costs. Length n_steps.
    :param A_ls: List of drift operators. Length n_steps.
    :param B_ls: List of control operators. Length n_steps.
    :return: Solved quadratic program for the dynamics.
    """
    # Shapes
    u_dim, n_steps = Ub.shape
    x_dim, _ = Xb.shape

    # Variables
    X = cp.Variable((x_dim, n_steps + 1), complex=True)
    U = cp.Variable([u_dim, n_steps])

    # QP
    cost = 0
    constr = []
    for t in range(n_steps):
        cost += cp.quad_form(X[:, t] - Xb[:, t], Q_ls[t]) + cp.quad_form(U[:, t] - Ub[:, t], R_ls[t])
        constr += [X[:, t + 1] == A_ls[t] @ X[:, t] + B_ls[t] @ U[:, t]]
    constr += [X[:, 0] == x0]
    cost += cp.quad_form(X[:, -1] - Xb[:, -1], Q_ls[-1])
    prob = cp.Problem(cp.Minimize(cp.real(cost)), constr)
    prob.solve(solver=cp.ECOS)
    return prob


def mpc():
    pass

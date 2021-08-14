import numpy as np
from numpy.linalg import multi_dot, pinv
# TODO: A challenge with LQR is limiting the control gradients.


def _dag(A):
    return A.conj().T


def _quad_form(x, Q):
    return multi_dot([_dag(x), Q, x])[0, 0].real


def quad_program(x0, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, u_prev=None, sat=None, verbose=False):
    # Shapes
    dim_u, horizon = U_bm.shape
    dim_x, _ = X_bm.shape

    # Gains
    Gains = [None] * horizon

    # Augments
    Q_aug = [None] * (horizon + 1)
    A_aug = [None] * horizon
    B_aug = [None] * horizon

    # Final value: time == horizon (only Q and X_bm)
    X_bm_t = X_bm[:, -1].reshape(-1, 1)
    qxt = Q_ls[-1] @ X_bm_t
    Q_aug[-1] = np.block([
        [Q_ls[-1], -qxt],
        [-_dag(qxt), _dag(X_bm_t) @ qxt]
    ])
    V = Q_aug[-1]
    # Run backup value iterations
    for t in reversed(range(horizon)):
        next_V = V

        # Reshape current benchmarks
        X_bm_t = X_bm[:, t].reshape(-1, 1)
        U_bm_t = U_bm[:, t].reshape(-1, 1)

        # Augment dynamics
        A_aug[t] = np.block([
            [A_ls[t], (A_ls[t] - np.identity(dim_x)) @ X_bm_t + B_ls[t] @ U_bm_t],
            [np.zeros([1, dim_x]), 1]
        ])
        B_aug[t] = np.block([
            [B_ls[t]],
            [np.zeros([1, dim_u])]
        ])

        # Augment state cost
        qxt = Q_ls[t] @ X_bm_t
        Q_aug[t] = np.block([
            [Q_ls[t], -qxt],
            [-_dag(qxt), _dag(X_bm_t) @ qxt]
        ])

        # Compute and save the current gain
        Gains[t] = -multi_dot([pinv(R_ls[t] + multi_dot([_dag(B_aug[t]), next_V, B_aug[t]])), _dag(B_aug[t]), next_V,
                               A_aug[t]])
        # Update the current value
        S = A_aug[t] + B_aug[t] @ Gains[t]
        V = Q_aug[t] + multi_dot([_dag(Gains[t]), R_ls[t], Gains[t]]) + multi_dot([S.conj().T, next_V, S])

    X0 = x0.reshape(-1, 1)
    U_opt = [None] * horizon
    X_opt = [None] * (horizon + 1)
    X_opt[0] = X0
    cost = 0
    for t in range(horizon):
        dX = X_opt[t] - X_bm[:, t].reshape(-1, 1)
        U_opt[t] = Gains[t] @ np.vstack([dX, 1]) + U_bm[:, t].reshape(-1, 1)
        U_opt[t] = np.clip(U_opt[t].real, -sat, sat)
        X_opt[t + 1] = A_ls[t] @ X_opt[t] + B_ls[t] @ U_opt[t]
        cost = cost + _quad_form(X_opt[t + 1], Q_ls[t + 1]) + _quad_form(U_opt[t], R_ls[t])
    return np.hstack(X_opt), np.hstack(U_opt), cost, Gains

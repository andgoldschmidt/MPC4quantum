import cvxpy as cp
from .experiment import isqrt
import numpy as np
from scipy.sparse import block_diag
from scipy.linalg import sqrtm

# Notes:
# 1. Final Q must be nonzero in CVXPY.
# 2. CVXPY uses column-major for reshape/vec, Numpy uses row-major.


def quad_program(x_init, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, Delta_ls, u_prev=None, sat=None, du=None, verbose=False):
    # Shapes
    dim_u, horizon = U_bm.shape
    dim_x, _ = X_bm.shape

    # Parameter
    X0 = cp.Parameter(dim_x, complex=True)

    # Variables
    X = cp.Variable((dim_x, horizon + 1), complex=True)
    U = cp.Variable([dim_u, horizon])

    # Init QP
    objective = 0
    constraints = []
    constraints += [X[:, 0] == X0]
    # Pin the initial control to u_prev (regularize freq.)
    if u_prev is not None and du is not None:
        constraints += [-du + u_prev.flatten() <= U[:, 0], U[:, 0] <= du + u_prev.flatten()]

    # Horizon
    for t in range(horizon):
        objective += cp.real(cp.quad_form(X[:, t] - X_bm[:, t], Q_ls[t]))
        objective += cp.real(cp.quad_form(U[:, t] - U_bm[:, t], R_ls[t]))
        # Taylor series:
        # dx/dt = f(x, u)
        #       = f(xg, ug) + df/dx(xg, ug) (x - xg) + df/du(xg, ug) (u - ug) + higher order
        #       = (f(xg, ug) - A xg - B ug) + A x + B u + higher order
        # Delta := (f(xg, ug) - A xg - B ug)
        constraints += [X[:, t + 1] == Delta_ls[t].flatten() + A_ls[t] @ X[:, t] + B_ls[t] @ U[:, t]]
        # Bound control
        constraints += [-sat * np.ones(dim_u) <= U[:, t], U[:, t] <= sat * np.ones(dim_u)]
        # Bound state?
        # xmin = np.zeros(dim_x)
        # xmax = np.ones(dim_x)
        # constraints += [-np.ones(dim_x) <= cp.real(X[:, t]), cp.real(X[:, t]) <= xmax]

        # Bound control du?
        # if du is not None and t > 1:
        #     constraints += [-du <= U[:, t] - U[:, t - 1],  U[:, t] - U[:, t - 1] <= du]

    # Final cost
    objective += cp.quad_form(X[:, -1] - X_bm[:, -1], Q_ls[-1])

    # Solve
    X0.value = x_init
    prob = cp.Problem(cp.Minimize(cp.real(objective)), constraints)
    obj_val = prob.solve(solver=cp.OSQP, verbose=verbose)
    return X.value, U.value, obj_val, prob


# def quad_program(x0, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, Delta_ls, u_prev=None, sat=None, du=None, verbose=False):
#     # Tutorial version
#     # Shapes
#     dim_u, horizon = U_bm.shape
#     dim_x, _ = X_bm.shape
#
#     # Variables
#     X = cp.Variable((dim_x, horizon + 1), complex=True)
#     U = cp.Variable([dim_u, horizon])
#
#     # Init QP
#     cost = 0
#     constr = []
#     constr += [X[:, 0] == x0]
#     # Pin the initial control to u_prev (regularize gradients)
#     if u_prev is not None and du is not None:
#         constr += [cp.norm(U[:, 0] - u_prev.flatten(), 'inf') <= du]
#
#     # Horizon
#     for t in range(horizon):
#         cost += cp.real(cp.quad_form(X[:, t] - X_bm[:, t], Q_ls[t]))
#         cost += cp.real(cp.quad_form(U[:, t] - U_bm[:, t], R_ls[t]))
#         # Taylor series:
#         # dx/dt = f(x, u)
#         #       = f(xg, ug) + df/dx(xg, ug) (x - xg) + df/du(xg, ug) (u - ug) + higher order
#         #       = (f(xg, ug) - A xg - B ug) + A x + B u + higher order
#         # Delta := (f(xg, ug) - A xg - B ug)
#         constr += [X[:, t + 1] == Delta_ls[t].flatten() + A_ls[t] @ X[:, t] + B_ls[t] @ U[:, t]]
#         # Control constraints
#         constr += [cp.norm(U[:, t], 2) <= sat]
#         if du is not None and t > 1:
#             constr += [cp.norm(U[:, t] - U[:, t - 1], 'inf') <= du]
#         # Density matrix constraints?
#         # sqrt_x = isqrt(dim_x)
#         # rho_t1 = cp.reshape(X[:, t + 1], (sqrt_x, sqrt_x), order='C')
#         # constr += [rho_t1 >> 0]
#         # constr += [cp.real(cp.trace(rho_t1)) <= 1]
#
#     # Final cost
#     cost += cp.quad_form(X[:, -1] - X_bm[:, -1], Q_ls[-1])
#
#     # Solve
#     prob = cp.Problem(cp.Minimize(cp.real(cost)), constr)
#     obj_val = prob.solve(solver=cp.ECOS, verbose=verbose)
#     return X.value, U.value, obj_val, prob


# def quad_program(x0, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, u_prev=None, sat=None, du=None, verbose=False):
#     """
#     Attempt to speed up the original quad program.
#
#     Note the following error:
#         ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has
#         1 dimension(s) and the array at index 1 has 2 dimension(s)
#
#     If you see this error, then cvxpy is unable to handle the fact that Q_ls[i] may be zero. This error appears when
#     converting the quadratic form to a norm which is what conic solvers do.
#     """
#     # Shapes
#     dim_u, horizon = U_bm.shape
#     dim_x, _ = X_bm.shape
#
#     # Flat variables
#     X = cp.Variable((dim_x * (horizon + 1)), complex=True)
#     U = cp.Variable((dim_u * horizon))
#
#     # Cost
#     cost = 0
#     R = block_diag(R_ls)
#     cost += cp.quad_form(U - U_bm.T.flatten(), R.toarray())
#     # note: conic solvers use norms not quadratic forms (manually replace <x,Qx> bc Q >= 0 can have bad numerics)
#     sqrt_Q = block_diag([sqrtm(q) for q in Q_ls]).tocsr()
#     cost += cp.norm(sqrt_Q.dot(X - X_bm.T.flatten()), 2)**2
#
#     # Dynamics constraints
#     # note: individual constraints for each time improve optimization success (empirical claim);
#     #       this is in contrast to a single RHS == LHS for all times.
#     # Taylor series:
#     # dx/dt = f(x, u) \approx f(xg, ug) + df/dx(xg, ug) (x - xg) + df/du(xg, ug) (u - ug)
#     constr = []
#     constr += [X[:dim_x] == x0]
#     for t in range(horizon):
#         RHS = A_ls[t] @ X[t * dim_x: (t + 1) * dim_x] + B_ls[t] @ (U[t * dim_u: (t + 1) * dim_u] - U_bm[:, t])
#         LHS = X[(t + 1) * dim_x: (t + 2) * dim_x]
#         constr += [LHS == RHS]
#         # constr += [cp.norm(LHS - RHS, 'inf') <= 1e-3]
#         if t > 1 and du is not None:
#             constr += [cp.norm(U[dim_u * t:dim_u * (t+1)] - U[dim_u * (t-1):dim_u * t], 'inf') <= du]
#     # Control constraint
#     # TODO: Needs better design!
#     if sat is not None:
#         constr += [cp.norm(U, 'inf') <= sat]
#     if u_prev is not None and du is not None:
#         constr += [cp.norm(U[:dim_u] - u_prev.flatten(), 'inf') <= du]
#
#     prob = cp.Problem(cp.Minimize(cp.real(cost)), constr)
#     obj_val = prob.solve()  # solver=cp.ECOS, verbose=verbose)
#     X_reshape = X.value if X.value is None else X.value.reshape(horizon + 1, dim_x).T
#     U_reshape = U.value if U.value is None else U.value.reshape(horizon, dim_u).T
#     return X_reshape, U_reshape, obj_val, prob

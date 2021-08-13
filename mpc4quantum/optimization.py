import cvxpy as cp
from scipy.sparse import block_diag
from scipy.linalg import sqrtm


# def quad_program(x0, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, u_prev=None, sat=None, verbose=False):
#     # Tutorial version
#     # Shapes
#     dim_u, n_steps = U_bm.shape
#     dim_x, _ = X_bm.shape
#
#     # Variables
#     X = cp.Variable((dim_x, n_steps + 1), complex=True)
#     U = cp.Variable([dim_u, n_steps])
#
#     # Init QP
#     cost = 0
#     constr = []
#     constr += [X[:, 0] == x0]
#
#     # QP
#     for t in range(n_steps):
#         cost += cp.quad_form(X[:, t] - X_bm[:, t], Q_ls[t]) + cp.quad_form(U[:, t] - U_bm[:, t], R_ls[t])
#         constr += [X[:, t + 1] == A_ls[t] @ X[:, t] + B_ls[t] @ U[:, t]]
#         constr += [cp.norm(U[:, t], 'inf') <= sat]
#     # Catch small bug in cvxpy related to zero quad form
#     if np.any(Q_ls[-1] > 0):
#         cost += cp.quad_form(X[:, -1] - X_bm[:, -1], Q_ls[-1])
#
#     # Pin control to u_prev (regularize gradients)
#     if u_prev is not None:
#         constr += [cp.norm(U[:, 0] - u_prev.flatten(), 1) <= 1]
#
#     prob = cp.Problem(cp.Minimize(cp.real(cost)), constr)
#     obj_val = prob.solve(solver=cp.ECOS, verbose=verbose)
#     return X.value, U.value, obj_val, prob


def quad_program(x0, X_bm, U_bm, Q_ls, R_ls, A_ls, B_ls, u_prev=None, sat=None, verbose=False):
    """
    Note the following error:
        ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has
        1 dimension(s) and the array at index 1 has 2 dimension(s)

    If you see this error, then cvxpy is unable to handle the fact that Q_ls[i] may be zero. This error appears when
    converting the quadratic form to a norm which is what conic solvers do.
    """
    # Shapes
    dim_u, n_steps = U_bm.shape
    dim_x, _ = X_bm.shape

    # Flat variables
    X = cp.Variable((dim_x * (n_steps + 1)), complex=True)
    U = cp.Variable((dim_u * n_steps))

    # Cost
    cost = 0
    R = block_diag(R_ls)
    cost += cp.quad_form(U - U_bm.T.flatten(), R)
    # note: conic solvers use norms not quadratic forms (manually replace <x,Qx> bc Q >= 0 can have bad numerics)
    sqrt_Q = block_diag([sqrtm(q) for q in Q_ls])
    cost += cp.norm(sqrt_Q @ (X - X_bm.T.flatten()), 2)**2

    # Dynamics constraints
    # note: individual constraints for each time improve optimization success (empirical claim);
    #       this is in contrast to a single RHS == LHS for all times.
    constr = []
    constr += [X[:dim_x] == x0]
    for t in range(n_steps):
        RHS = A_ls[t] @ X[t * dim_x: (t + 1) * dim_x] + B_ls[t] @ U[t * dim_u: (t + 1) * dim_u]
        LHS = X[(t + 1) * dim_x: (t + 2) * dim_x]
        constr += [LHS == RHS]
        # constr += [cp.norm(LHS - RHS, 'inf') <= 1e-4]

    # Control constraint
    # TODO: Needs better design!
    if sat is not None:
        constr += [cp.norm(U, 'inf') <= sat]
    if u_prev is not None:
        constr += [cp.norm(U[:dim_u] - u_prev.flatten(), 1) <= 0.5]

    prob = cp.Problem(cp.Minimize(cp.real(cost)), constr)
    obj_val = prob.solve(solver=cp.ECOS, verbose=verbose)
    X_reshape = X.value if X.value is None else X.value.reshape(n_steps + 1, dim_x).T
    U_reshape = U.value if U.value is None else U.value.reshape(n_steps, dim_u).T
    return X_reshape, U_reshape, obj_val, prob

import numpy as np
from itertools import combinations

# Class to linearize nonlinear dynamics.
# TODO: Is the Taylor expansion along the discrete operators appropriate? Compare to continuous.


class WrapModel:
    """
    Assumes a model with a polynomial control library up to a given order. There are likely other assumptions
    here. The use case is the local approximation of a bilinear model for NMPC.
    """
    def __init__(self, A_op, N_op, dim_u, order):
        # A @ x + N @ (f(u) * x)
        self.A = A_op
        self.N = N_op

        # Get dimensions
        self.dim_x = self.A.shape[1]
        self.dim_u = dim_u
        self.order = order
        self.polyu_dim = int(self.N.shape[1] / self.dim_x)
        if not np.isclose(size_of_library(self.order, self.dim_u) - 1, self.polyu_dim):
            raise ValueError("Dimension mismatch when wrapping a model operator.")

        # Get library tools
        self.fns = create_library(self.order, self.dim_u)[1:]
        self.deriv_fns, self.deriv_coefs = diff_library(self.order, self.dim_u)

        # N's actions must be unpacked in order to accommodate dN/dx or dN/du.
        # Assumes that the control operator was constructed with [y_dim, krtimes(polyu_dim, dim_x)]
        self.unpacked_N = N_op.reshape(self.dim_x, self.polyu_dim, self.dim_x)

    def lift_u(self, u_shaped):
        return np.vstack([f(u_shaped) for f in self.fns])

    def f(self, x, u, t):
        x_shaped = x.reshape(-1, 1)
        u_shaped = u.reshape(-1, 1)
        polyu = self.lift_u(u_shaped)
        return self.A @ x_shaped + self.N @ krtimes(polyu, x_shaped)

    def df_dx(self, x, u, t):
        u_shaped = u.reshape(-1, 1)
        # Get B(u) acting on x
        polyu = self.lift_u(u_shaped)
        B = np.hstack([self.unpacked_N[:, :, i] @ polyu for i in range(self.dim_x)])
        return self.A + B

    def df_du(self, x, u, t):
        x_shaped = x.reshape(-1, 1)
        u_shaped = u.reshape(-1, 1)
        # Get B(x) acting on f(u)
        polyB = np.hstack([self.unpacked_N[:, i, :] @ x_shaped for i in range(self.polyu_dim)])
        # Compute an operator for each coordinate of u. Each result is a column. Stack to act on u.
        B = []
        for i in range(self.dim_u):
            B.append(polyB @ (self.deriv_coefs[i] * np.vstack([f(u_shaped) for f in self.deriv_fns[i]])))
        return np.hstack(B)

    def get_model_along_traj(self, xs, us, ts):
        A = []
        B = []
        Delta = []
        for i in range(len(ts)):
            A.append(self.df_dx(xs[:, i], us[:, i], ts[i]))
            B.append(self.df_du(xs[:, i], us[:, i], ts[i]))
            Pred = A[-1] @ xs[:, i] + B[-1] @ us[:, i]
            Delta.append(self.f(xs[:, i], us[:, i], ts[i]) - Pred[:, None])
        return A, B, Delta

    def get_model_from_initial(self, xs, us, ts):
        A = [self.df_dx(xs[:, 0], us[:, 0], ts[0])] * len(ts)
        B = [self.df_du(xs[:, 0], us[:, 0], ts[0])] * len(ts)
        Pred = A[0] @ xs[:, 0] + B[0] @ us[:, 0]
        Delta = [self.f(xs[:, 0], us[:, 0], ts[0]) - Pred[:, None]] * len(ts)
        return A, B, Delta


def krtimes(A, B):
    # Khatri-Rao product
    At = A.T
    Bt = B.T
    na = At.shape[0]
    nb = Bt.shape[0]
    if not na == nb:
        raise ValueError("Cols of A =/ Cols of B")
    else:
        return np.einsum('ij,ik->ijk', At, Bt).reshape(na, -1).T


def multinomial_powers(n, k):
    """
    Returns all combinations of powers of the expansion (x_1+x_2+...+x_k)^n.
    The motivation for the algorithm is to use dots and bars:
    e.g.    For (x1+x2+x3)^3, count n=3 dots and k-1=2 bars.
            ..|.| = [x1^2, x2^1, x3^0]

    Note: Add 1 to k to include a constant term, (1+x+y+z)^n, to get all
    groups of powers less than or equal to n (just ignore elem[0])

    Emphasis is on preserving yield behavior of combinatorial iterator.

    Arguments:
        n: the order of the multinomial_powers
        k: the number of variables {x_i}
    """
    for elem in combinations(np.arange(n + k - 1), k - 1):
        elem = np.array([-1] + list(elem) + [n + k - 1])
        yield elem[1:] - elem[:-1] - 1


def create_power_list(order, dimension):
    return [p[:-1] for p in multinomial_powers(order, dimension + 1)]


def size_of_library(order, dimension):
    return len(create_power_list(order, dimension))


def create_library_from_list(power_list):
    library = []
    for powers in np.array(power_list):
        library.append(lambda x, ps=powers: np.product([np.zeros_like(x[i, :]) if p < 0 else np.power(x[i, :], p)
                                                        for i, p in enumerate(ps)], axis=0))
    return library


def create_library(order, dimension):
    """
    Returns a library of functions up to 'order' that apply to variables in
    a space of 'dimension'.

    Arguments:
        order: The maximal order of multinomials to construct in the library.
        dimension: The number of dimensions in the input space.
    """
    return create_library_from_list(create_power_list(order, dimension))


def diff_library(order, dimension):
    plist = np.vstack(create_power_list(order, dimension)[1:])
    dlist = create_power_list(1, dimension)[1:]
    deriv_powers = [None] * len(dlist)
    deriv_coef = [None] * len(dlist)
    for i, d in enumerate(dlist):
        deriv_powers[i] = plist - d
        deriv_coef[i] = plist[:, d.astype(bool)]
    return [create_library_from_list(p) for p in deriv_powers], deriv_coef

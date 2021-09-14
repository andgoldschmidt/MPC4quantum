import numpy as np
import qutip as qt
import itertools as it

from .linearize import create_power_list


def discretize_homogeneous(A_cts_list, dt, order):
    # Assume square ops
    dim_x = A_cts_list[0].shape[0]
    id_op = np.identity(dim_x)

    # Construct the powers that match the control + state vector
    dim_u = len(A_cts_list) - 1
    powers_list = create_power_list(order, dim_u)
    A_dst_list = [np.zeros((dim_x, dim_x), dtype=complex)] * len(powers_list)

    # Iterate over expansion order
    for an_order in range(0, order + 1):
        prefactor = 1 / np.math.factorial(an_order) * dt ** an_order
        # Compute all non-commutative products in sum ** an_order
        for a_product in it.product(range(len(A_cts_list)), repeat=an_order):
            entry = id_op
            for i_op in a_product:
                entry = entry @ A_cts_list[i_op]
            entry = prefactor * entry

            # Attach entry to the correct part of the discrete operator, based on the control + state vector
            elems, counts = np.unique(a_product, return_counts=True)
            powers = np.zeros(dim_u + 1)
            powers[elems.astype(int)] = counts
            match = np.argwhere(np.product(powers_list == np.array(powers[1:]), axis=1))
            if len(match) != 1:
                raise ValueError('Error in discretization. Control powers should contribute uniquely.')
            index = match[0, 0]
            A_dst_list[index] = A_dst_list[index] + entry
    return np.hstack(A_dst_list)


def vectorize_me(H, measure_list):
    dim_m = len(measure_list)

    # Precompute structure constants of measurement basis
    structure_table = []
    for i, sigma_i in enumerate(measure_list):
        for j, sigma_j in enumerate(measure_list):
            for k, sigma_k in enumerate(measure_list):
                struct_const = 0. if i == j else (qt.commutator(sigma_i, sigma_j).dag() * sigma_k).tr()
                structure_table.append([i, j, k, struct_const])
    structure_table = np.array(structure_table).reshape(dim_m, dim_m, dim_m, -1)[:, :, :, -1]

    # Project hamiltonian
    H_list = [(H.dag() * sigma_i).tr() for sigma_i in measure_list]

    # Project Liouville equation
    A_op = []
    for k in range(dim_m):
        for j in range(dim_m):
            entry = 0
            for i in range(dim_m):
                entry = entry + H_list[i] * structure_table[i, k, j]
            A_op.append([j, k, -1j * entry])
    return np.array(A_op).reshape(dim_m, dim_m, -1)[:, :, -1]

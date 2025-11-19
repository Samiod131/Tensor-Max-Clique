import warnings
import time

import numpy as np
import networkx as nx

import TNTools as tnt


'''
Max-Clique finder using TNTools MPS-MPO formalism.
'''



def initiate_clique_mps(n, p):
    '''
    The mps for the max clique problem solving. P is the exponent constant
    associated with each clique size.
    '''
    # initiate tensor
    tens = np.array([1, p]).reshape(1, 2, 1)

    # Append for all mps tensors
    mps = []
    for _ in range(0, n):
        mps.append(tens)

    return mps


'''
################################################################################
MPO generation and adjacency matrix
################################################################################
'''


def _split_nand():
    s_nand = np.array([[1, 1, 1, 1], [1, 0, 0, 0]]).reshape(2, 2, 2)
    return s_nand


def _one_site_nand():
    s_nand = np.array([[1, 1], [1, 0]])
    return s_nand


def _or_top_tens():

    left_or = np.zeros([2, 2, 2])
    left_or[:, :, 1] = np.array([[0, 1], [1, 1]])
    left_or[:, :, 0] = np.array([[1, 0], [0, 0]])

    return left_or


def _or_bottom_tens():

    right_or = np.zeros([2, 2, 2])
    right_or[:, 1, :] = np.array([[0, 1], [1, 1]])
    right_or[:, 0, :] = np.array([[1, 0], [0, 0]])

    return right_or


def max_clique_constraints(bites_array, vertex_pos, adjacency=True):
    '''
    Generates a mpo for one column of the adjacency matrix of a graph. Builds
    the mpo constraint associated with max clique problem.
    '''
    # Creating simple cross tensor
    cross = np.tensordot(np.identity(2), np.identity(2), axes=0)

    # Initiate all or tensors positions
    if adjacency is True:
        # Exclude the main vertex position
        bites_array[vertex_pos] = 1
        # When receiving adj matrix column, find positions of all or tensors
        tens_pos = np.ones(len(bites_array))-bites_array
        tens_pos = np.nonzero(tens_pos)[0]
    else:
        tens_pos = np.array([t for t in bites_array])

    mpo = []
    start = None
    end = None
    # Exception for a vertex with no connections
    if tens_pos.size > 0:
        # Define the starting and ending point of the mpo
        start = tens_pos[0]
        end = tens_pos[-1]
        # Adapt start and end position depending of the control vertex's position
        if vertex_pos < start:
            start = vertex_pos
        if vertex_pos > end:
            end = vertex_pos

        for i in range(start, end+1):
            # elements before the main vertex position
            if i < vertex_pos:
                if i == start:
                    # First and lasttensor is simply a copy tensor passing value
                    if i in tens_pos:
                        mpo_t = tnt.const_mpo_tens(
                            np.array([[1, 0], [0, 1]]), edge="start")
                    else:
                        raise ValueError('First tensor position is trivial.')

                else:
                    if i in tens_pos:
                        # Special tensor is used (refer to thesis)
                        mpo_t = tnt.const_mpo_tens(_or_top_tens())
                    else:
                        mpo_t = cross

            # for after the main vertex
            elif i > vertex_pos:
                if i == end:
                    if i in tens_pos:
                        # First and lasttensor is simply a copy tensor passing value
                        mpo_t = tnt.const_mpo_tens(
                            np.array([[1, 0], [0, 1]]), edge="end")
                    else:
                        raise ValueError('Last tensor position is trivial.')

                else:
                    if i in tens_pos:
                        # Special tensor is used (refer to thesis)
                        mpo_t = tnt.const_mpo_tens(_or_bottom_tens())
                    else:
                        mpo_t = cross
            # If it is the main vertex
            elif i == vertex_pos:
                # If it is edges, more simple element
                if i == start:
                    mpo_t = tnt.const_mpo_tens(_one_site_nand(), edge="start")
                elif i == end:
                    mpo_t = tnt.const_mpo_tens(_one_site_nand(), edge="end")
                # Otherwise, more general tensor
                else:
                    mpo_t = tnt.const_mpo_tens(_split_nand())

            mpo.append(mpo_t)

    return mpo, start, end


'''
################################################################################
Contractions and solving
################################################################################
'''


def max_clique_solver(adj_matrix, const=2, max_chi=False, dmrg_chi=False, check_entropy=False):
    '''
    Find the maximum clique of a graph by receiving its adjacency matrix.
    Returns the largest clique's vertices coordinates.
    '''

    def svd_func(_m):
        return tnt.reduced_svd(_m, max_len=max_chi, err_th=1e-10)

    n = adj_matrix.shape[0]
    mps = initiate_clique_mps(n=n, p=const)
    mps_class = tnt.MpsStateCanon(mps, svd_func=svd_func)
    mps_class.create_orth()

    max_bond = mps_class.max_bond_dim()
    if check_entropy is False:
        max_entropy = False
    else:
        max_entropy = mps_class.entang_entropy()

    for main_vertex in range(0, n):
        column = adj_matrix[:, main_vertex]
        if column[main_vertex] != 0:
            raise ValueError(
                'Diagonal elements adj matrix must be of value zero.')

        mpo, start, _ = max_clique_constraints(column, main_vertex)

        if len(mpo) != 0:
            try:
                mps_class.mpo_contract(mpo=mpo, beginning=start)
            except ValueError:
                warnings.warn(
                    'Total state elimination. Try again with higher bond dimension.')
                return [0], max_bond, max_entropy

            bond = mps_class.max_bond_dim()
            if bond > max_bond:
                max_bond = bond

            if check_entropy is True:
                entropy = mps_class.entang_entropy()
                if entropy > max_entropy:
                    max_entropy = entropy

    # Avoiding ARPACK eigs convergence problem.
    trial = 0
    converge = False
    while (trial < 5) and (converge is False):
        try:
            found_main = mps_class.main_component(
                method='dephased_DMRG', chi_max=dmrg_chi, energy_var=0)
            found_main = list(np.nonzero(found_main)[0])
            # Verify the result
            converge = True
        except:
            trial += 1
            found_main = [0]

    return found_main, max_bond, max_entropy


import warnings
import time

import numpy as np
import networkx as nx

import TNTools as tnt


'''
Max-Clique finder using TNTools MPS-MPO formalism.
'''

'''
################################################################################
Data saving and others
################################################################################
'''


def save_to_file(parameters, filename, header=False, path='./', buffer=20):

    # Check if result file already exist, and initiate it if not
    try:
        with open(path+filename) as _:
            pass

    except FileNotFoundError:
        if header is False:
            print('The file doesn\'t exist. Creating one for data.')

        else:
            print('File doesn\'t exist, Creating one with given header.')
            header_line = ''
            # enumerate all header elements and convert to a atring line
            for i, param in enumerate(header):
                # If parameter is last of list, do not separate w. comma & buffer
                if i == len(parameters)-1:
                    param_add = str(param)
                else:
                    param_add = str(param)+','

                # Put space for regular column look
                skip = buffer-len(param_add)

                # If value string larger than buffer. Just add full buffer value.
                if skip <= 0:
                    param_add += ' '*buffer
                else:
                    param_add += ' '*skip

                # Add parameter string to the whole lign
                header_line += param_add

            # Print string line on file and close it
            save = open(path+filename, 'a')
            print(header_line, file=save)
            save.close()

    step_line = ''
    # see header printing fo context on top
    for i, param in enumerate(parameters):
        if i == len(parameters)-1:
            param_add = str(param)
        else:
            param_add = str(param)+','
        skip = buffer-len(param_add)
        if skip <= 0:
            param_add += ' '*buffer
        else:
            param_add += ' '*skip
        step_line += param_add

    save = open(path+filename, 'a')
    print(step_line, file=save)
    save.close()


'''
################################################################################
Graph generation and adjacency matrix extraction
################################################################################
'''


def find_all_max_cliques(g):
    '''
    Find all maximum cliques of a given graph.
    '''
    # find all maximal cliques
    cliques = nx.find_cliques(g)

    # Iterate over all cliques and only keep the largest ones.
    max_size = 0
    max_cliques = []
    for _, clique in enumerate(cliques):
        c_size = len(clique)
        if c_size > max_size:
            max_size = c_size
            max_cliques = []
            clique.sort()
            max_cliques.append(clique)
        elif c_size == max_size:
            clique.sort()
            max_cliques.append(clique)

    return max_cliques


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
MPS generation and contractions
################################################################################
'''


def bitflip_array(p, n):
    '''
    Return 0-array of n bits with fixed bit flip probability p
    '''
    arr = np.random.choice([0, 1], n, p=[1-p, p])
    return arr


def initiate_clique_mps(n, p):
    '''
    The mps for the max clique problm solving. P is the exponent constant
    associated with each clique size.
    '''
    # initiate tensor
    tens = np.array([1, p]).reshape(1, 2, 1)

    # Append for all mps tensors
    mps = []
    for _ in range(0, n):
        mps.append(tens)

    return mps


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


def single_max_clique_test(n, p, chi=False, dmrg_chi=False, const=2, debug=False):
    # Generate random NetworkX graph and get adj matrix
    g = nx.fast_gnp_random_graph(n=n, p=p)
    adj_mat = nx.to_numpy_array(g)

    exact_t = time.time()
    # Find all maximum cliques in a graph
    real_max_cliques = find_all_max_cliques(g)
    exact_t = time.time()-exact_t

    tens_t = time.time()
    # Apply method
    found_max, time_max_bond, max_entropy = max_clique_solver(
        adj_mat, max_chi=chi, dmrg_chi=dmrg_chi, const=const)
    tens_t = time.time()-tens_t

    if debug:
        print('Exact solver time: '+str(exact_t))
        print('Tensor time: '+str(tens_t))
        print('Real max cliques: '+str(real_max_cliques))
        print('Found max clique: '+str(found_max))

    failure = 1
    if found_max in real_max_cliques:
        failure = 0

    return failure, time_max_bond, max_entropy, tens_t


def single_error_rate(n, p, chi=False, dmrg_chi=False, const=2, sample=100):

    failures = []
    max_bonds = []
    max_entropy = []
    alg_time = []
    # Header list for saving file
    for _ in range(0, sample):

        fail, max_b, max_s, tens_time = single_max_clique_test(
            n=n, p=p, chi=chi, dmrg_chi=dmrg_chi, const=const)
        failures.append(fail)
        max_bonds.append(max_b)
        max_entropy.append(max_s)
        alg_time.append(tens_time)

    return failures, max_bonds, max_entropy, alg_time


def max_clique_full_search(sizes=[8], probs=[0.3], chis=[8],  sample=100, const=2, folder='./results/'):

    # Header list for saving file
    header = ['const', 'size', 'state_chi', 'dmrg_chi', 'prob', 'sample_size', 'fail_rate', 'time_max_bond', 'std_max_bond', 'max_entrop', 'std_max_entrop',
              'avg_time', 'std_time']

    # Looping over all possible parameters and printing
    for _c, chi in enumerate(chis):
        print('[Run for chi='+str(chi)+' bits ('+str(_c+1)+'/' +
              str(len(chis))+')]')
        for _s, size in enumerate(sizes):
            print(
                '[size='+str(size)+' bits ('+str(_s+1)+'/' + str(len(sizes))+')]')
            for _p, p in enumerate(probs):
                print('Probability of '+str(p) +
                      ' ('+str(_p+1)+'/'+str(len(probs))+')')

                d_chi = chi
                fail_r, max_bonds, entrop, times = single_error_rate(
                    n=size, p=p, chi=chi, dmrg_chi=d_chi, sample=sample, const=const)

                f_rate = np.average(fail_r)
                t_max_bond = np.average(max_bonds)
                std_max_b = np.std(max_bonds)
                if entrop[0] is False:
                    max_entrop = False
                    std_max_entrop = False
                else:
                    max_entrop = np.average(entrop)
                    std_max_entrop = np.std(entrop)
                avg_time = np.average(times)
                std_time = np.std(times)

                # Writing parameters list to file
                step_params = [const, size, chi, d_chi, p, sample, f_rate, t_max_bond, std_max_b,
                               max_entrop, std_max_entrop, avg_time, std_time]
                save_to_file(step_params, 'results_s_chi_'+str(chi)+'_N_'+str(size)+'_dchi_'+str(d_chi)+'.dat',
                             path=folder, header=header)
                print('Failure rate: '+str(f_rate))


'''
################################################################################
Testing algorithm
################################################################################
'''


sizes = [8]
p = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
     0.55,  0.6, 0.65, 0.7, 0.75,  0.8, 0.85,  0.9, 0.95, 1]
chi = [8, 16, 24, 32, 40, 48, 56, 64]
sample = 500

max_clique_full_search(sizes=sizes, probs=p, chis=chi, sample=sample)

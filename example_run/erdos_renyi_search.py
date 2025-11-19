import warnings
import time

import numpy as np
import networkx as nx

import tnt_max_clique as mc


'''
Example code for max clique problem solving for Erdős-Rényi graph solving.
'''

'''
################################################################################
Saving and others
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
Exact Validation method
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



#Testing and macro


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
    found_max, time_max_bond, max_entropy = mc.max_clique_solver(
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


if __name__ == "__main__":
    sizes = [8]
    p = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55,  0.6, 0.65, 0.7, 0.75,  0.8, 0.85,  0.9, 0.95, 1]
    chi = [8, 16]
    sample = 500

    max_clique_full_search(sizes=sizes, probs=p, chis=chi, sample=sample)

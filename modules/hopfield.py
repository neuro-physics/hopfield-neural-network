import numpy as np
from enum import IntEnum

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

def find_first(cond):
    k = np.nanargmax(cond)
    return k if cond[k] else -1

def generate_ising_states(N):
    """
    Generate the first 2^(N-1) spin configurations of an N-spin Ising system.

    This function constructs a matrix of Ising states, where each row represents 
    one possible configuration of N spins. Spins are encoded as ±1 values, 
    corresponding to "up" (+1) and "down" (-1). Only the first half of the 
    possible states (2^(N-1)) are generated, which is often sufficient due to 
    symmetry considerations in Ising models.

    Parameters
    ----------
    N : int
        The number of spins in the system. Determines both the number of 
        configurations generated and the dimensionality of each state vector.

    Returns
    -------
    ising_states : numpy.ndarray
        A matrix of shape (2^(N-1), N), where each row corresponds to a spin 
        configuration. Entries are ±1, representing spin orientations.

    Notes
    -----
    - The number of states is computed as `1 << (N - 1)`, which equals 2^(N-1).
    - Bitwise operations are used to efficiently extract spin orientations 
      from integer indices:
        * Each index encodes a spin configuration in binary form.
        * Right-shifting and bitwise AND yield individual spin values (0 or 1).
    - Values are immediately converted to `float64` to avoid overflow in 
      subsequent numerical operations.
    - Spin mapping: `0 → +1`, `1 → -1`.

    Examples
    --------
    >>> generate_ising_states(3)
    array([[ 1.,  1.,  1.],
           [-1.,  1.,  1.],
           [ 1., -1.,  1.],
           [-1., -1.,  1.]])

    >>> generate_ising_states(4).shape
    (8, 4)
    """
    num_states = 1 << (N - 1)
    indices = np.arange(num_states, dtype=np.uint64)
    # Extract bits: 0 or 1
    # We use bit shifting and bitwise AND to get 0s and 1s
    # shape will be (num_states, n)
    # Force the shift array to be the same type as indices
    # We explicitly convert the result of the bit-shift to a signed float64 
    # immediately to prevent overflow in subsequent math.
    states = ((indices[:, None] >> np.arange(N, dtype=np.uint64)) & 1).astype(np.float64)
    # Transform to +/- 1 spins
    # Mapping: 0 -> 1, 1 -> -1 (or vice versa)
    ising_states = 1.0 - 2.0 * states
    return ising_states

def find_memory_indices(states, memories):
    """
    Identify the positions in `states` where any given memory or its negation (anti-memory) occurs.

    Parameters
    ----------
    states : list of numpy.ndarray
        A sequence of state vectors to be checked.
    memories : list of numpy.ndarray
        A collection of memory vectors to search for within `states`.

    Returns
    -------
    list of int
        Indices in `states` where a state matches either a memory vector
        or its negation (-memory).
    
    Notes
    -----
    - A "memory" is defined as an exact match to one of the vectors in `memories`.
    - An "anti-memory" is defined as the negation of a memory vector.
    - Matching is performed using `numpy.array_equal`.
    """
    memory_indices = []
    for i, state in enumerate(states):
        for mem in memories:
            # Memory or anti-memory
            if np.array_equal(state, mem) or np.array_equal(state, -mem):
                memory_indices.append(i)
                break
    return np.array(memory_indices)

def sort_basins(state_mem, E):
    """
    Group states by attractor basin and sort the states inside each basin
    according to their energy.
    Parameters
    ----------
    state_mem : ndarray of shape (n_states,)
        Integer array whose element ``state_mem[i]`` gives the index mu 
        of the attractor (memory) reached by state ``i`` under the network dynamics.
    E : ndarray of shape (n_states,)
        Energy associated with each network state.
    Returns
    -------
    states_basin : ndarray of shape (n_states,)
        Array containing the reordered state indices. States are first
        grouped by attractor basin and then sorted by increasing energy
        within each basin.
    E_basin : ndarray of shape (n_states,)
        Energies reordered consistently with ``states_basin``.
    Notes
    -----
    The ordering produced by this function is useful for visualizing the
    energy landscape of Hopfield networks, since states belonging to the
    same attractor basin appear grouped together.
    """
    # first group states by attractor index
    states_basin = np.argsort(state_mem)
    # reordered energies
    E_basin = E[states_basin]
    # unique basin labels
    basins = np.unique(state_mem)
    # sort states inside each basin by energy
    for mu in basins:
        # positions inside the reordered array
        ind = np.where(state_mem[states_basin] == mu)[0]
        # energy ordering inside this basin
        k = np.argsort(E_basin[ind])
        # reorder basin states
        states_basin[ind] = states_basin[ind][k]
        E_basin[ind]      = E_basin[ind][k]
    return states_basin, E_basin

def sort_states_around_energy_minima(states, energies, memories):
    """
    Reorder states to highlight each memory as a local energy minimum.

    Parameters
    ----------
    states : list of numpy.ndarray
        Sequence of state vectors to be sorted.
    energies : list or numpy.ndarray
        Energy values corresponding to each state.
    memories : list of numpy.ndarray
        Memory vectors whose positions should act as minima.

    Returns
    -------
    states_sorted : numpy.ndarray
        A 2D array of states reordered so that each memory (or anti-memory)
        forms the center of a valley in the energy landscape.
    ind_sorted : numpy.ndarray
        sorted index of the states
    E_sorted : numpy.ndarray
        sorted energy of the states

    Strategy
    --------
    1. Locate indices where memories or anti-memories occur in `states`.
    2. Assign each state to the nearest memory index using Euclidean distance
       between state indices (not vector values).
    3. Within each memory basin:
       - States before the memory are sorted by descending energy.
       - States at or after the memory are sorted by ascending energy.
    4. Concatenate all basins to form the final ordering.

    Notes
    -----
    - If no memories are found, states are simply sorted by ascending energy.
    - This arrangement creates a valley-like structure centered on each memory.
    """
    find_memory    = lambda states,m: np.argmax([np.array_equal(s, m) for s in states])
    mem_ind        = [ find_memory(states,m) for m in memories ]
    antimem_ind    = [ find_memory(states,-m) for m in memories ]
    memory_indices = np.array(sorted(mem_ind+antimem_ind))
    # Fallback: no memories found
    if len(memory_indices) == 0:
        order = np.argsort(energies)
        return np.atleast_2d(states[order]), order, energies[order]
    # Group states by nearest memory
    groups = {m: [] for m in memory_indices}
    for i, state in enumerate(states):
        # index of nearest memory
        nearest = min(memory_indices, key=lambda m: abs(i - m))
        groups[nearest].append((i, energies[i], state))
    # Sort left/right states inside each groups
    sorted_ind    = []
    sorted_E      = []
    sorted_states = []
    for mem_idx in sorted(memory_indices):
        group = groups[mem_idx]
        # Left: high -> low energy
        left_sorted  = sorted([x for x in group if x[0] < mem_idx],
                              key=lambda x: x[1], reverse=True)
        # Right: low -> high energy
        right_sorted = sorted([x for x in group if x[0] >= mem_idx],
                              key=lambda x: x[1])
        for x in left_sorted + right_sorted:
            sorted_ind.append(x[0])
            sorted_E.append(x[1])
            sorted_states.append(x[2])
    return np.atleast_2d(sorted_states),np.array(sorted_ind),np.array(sorted_E)

def save_sorted_state_txt(fname,states_set,patterns,sorted_ind,states_sorted,states_basin_ind,states_basin):
    state_to_str    = lambda state: ''.join([ ('+' if s>0 else '-') for s in state ])
    all_memories = np.concatenate((np.array(patterns),-np.array(patterns)))
    mem_ind      = find_memory_indices(states_set,all_memories)

    txt  = np.array([ f'\t{n+1:5d}\t:\t{state_to_str(s):s}\t\t|\t{m+1:5d}\t:\t{state_to_str(x):s}\t\t|\t{k+1:5d}\t:\t{state_to_str(y):s}\t:\t{b+1:3d}' for n,(s,m,x,k,y,b) in enumerate(zip(states_set,sorted_ind,states_sorted,states_basin_ind,states_set[states_basin_ind],states_basin)) ], dtype=str)
    txtm = np.array([ f'\t{mu+1:5d}\t:\t{state_to_str(s):s}\t\t(n={n+1:5d})' for mu,(n,s) in enumerate(zip(mem_ind,all_memories)) ], dtype=str)

    txt = np.concatenate((
        ['# memories and anti-memories (xi)'],
        ['# \t\tmu\t:\tstate'],
        txtm,
        ['# original state ordering\t\t| sorted states (sigma)'],
        ['# \t\tn\t:\tstate\t\t\t|\t\tn\t:\tstate\t\t\t|\t\tn\t:\tstate\t\t:\tbasin'],
        txt
    ))
    np.savetxt(fname,txt,fmt='%s')
    print(f' *** file saved ... {fname}')
    return txt

class DistributionType(IntEnum):
    Uniform    = 1
    Normal     = 2
    IntUniform = 3

def generate_random_interaction_matrix(N, dist_type: DistributionType = 2):
    """
    Generate a symmetric random interaction matrix with zero diagonal entries.

    This function creates an NxN matrix representing pairwise interactions 
    between elements (e.g., nodes, spins, agents). The entries are drawn 
    from a specified probability distribution, then symmetrized to ensure 
    the matrix is symmetric. Self-interactions (diagonal entries) are set to zero.

    Parameters
    ----------
    N : int
        The size of the matrix (number of elements). The resulting matrix 
        will have shape (N, N).
    
    dist_type : DistributionType, optional
        The type of distribution to use for generating random values. 
        Supported options are:
        
        - `DistributionType.Uniform`: entries are drawn from a uniform 
          distribution over [0, 1).
        - `DistributionType.IntUniform`: entries are randomly chosen 
          as either -1 or +1 with equal probability.
        - `DistributionType.Normal`: entries are drawn from a standard 
          normal distribution (mean = 0, variance = 1).
        
        Default is `2`, which should correspond to one of the defined 
        `DistributionType` values.

    Returns
    -------
    J : numpy.ndarray
        A symmetric (N, N) matrix with zero diagonal entries. The off-diagonal 
        entries represent random interactions drawn from the specified distribution.

    Notes
    -----
    - Symmetry is enforced by averaging the matrix with its transpose:
      `(J + J.T) / 2.0`.
    - Diagonal entries are explicitly set to zero to remove self-interactions.
    - This function is useful in contexts such as statistical physics 
      (e.g., spin glass models), network theory, or simulations requiring 
      random symmetric interaction matrices.

    Examples
    --------
    >>> generate_random_interaction_matrix(4, DistributionType.Uniform)
    array([[ 0.        ,  0.312...,  0.456...,  0.789...],
           [ 0.312...,  0.        ,  0.654...,  0.123...],
           [ 0.456...,  0.654...,  0.        ,  0.987...],
           [ 0.789...,  0.123...,  0.987...,  0.        ]])

    >>> generate_random_interaction_matrix(3, DistributionType.IntUniform)
    array([[ 0., -1.,  1.],
           [-1.,  0., -1.],
           [ 1., -1.,  0.]])
    """
    if dist_type == DistributionType.Uniform:
        J = np.random.rand(N,N)
    elif dist_type == DistributionType.IntUniform:
        J = (2.0*np.random.randint(0,2,size=(N,N))-1.0).astype(float)
    elif dist_type == DistributionType.Normal:
        J = np.random.normal(0, 1, (N, N))
    J = (J + J.T) / 2.0  # Ensure symmetry
    np.fill_diagonal(J, 0) # Remove self-interactions
    return J


def get_H_pattern(L=10):
    """Generates a 2D 'H' shape pattern flattened into a vector."""
    pattern          = -np.ones((L, L))
    pattern[1:-1, 2] = 1   # Left bar
    pattern[1:-1, 7] = 1   # Right bar
    pattern[5, 2:8]  = 1    # Crossbar
    return pattern.flatten()

def get_X_pattern(L=10):
    """Generates a 2D 'X' shape pattern as a second memory example."""
    pattern = -np.ones((L, L))
    for i in range(L):
        pattern[i, i]         = 1
        pattern[i, L - 1 - i] = 1
    return pattern.flatten()

def get_plus_pattern(L=10):
    """Generates a 2D '+' shape pattern"""
    p                 = -np.ones((L,L))
    m                 = L//2
    p[:,m]            = 1
    p[m,:]            = 1
    p[:,max((m-1,0))] = 1
    p[max((m-1,0)),:] = 1
    return p.flatten()

def get_stripe_pattern(L=10):
    """Generates a 2D '+' shape pattern"""
    p      = -np.ones((L,L))
    p[:,L//3]   = 1
    p[L//3,:]   = 1
    p[:,2*L//3] = 1
    p[2*L//3,:] = 1
    return p.flatten()

def _make_list(X):
    if type(X) is not list:
        return [X]
    else:
        return X

def add_noise(pattern, noise_level=0.25):
    """Randomly flips bits in a pattern."""
    s_noisy                = pattern.copy()
    N                      = len(pattern)
    flip_indices           = np.random.choice(N, size=int(noise_level * N), replace=False)
    s_noisy[flip_indices] *= -1
    return s_noisy

def  _exists(X):
    return not(type(X) is type(None))

def _is_nparray_of_object(X):
    return isinstance(X,np.ndarray) and (X.dtype == np.object_)

def _is_valid_list_of_img(img_lst):
    return isinstance(img_lst,list) or _is_nparray_of_object(img_lst)

def _resize_img_cv2(img,L):
    if cv2:
        h,w = img.shape[:2]
        if h >= w:
            img_s = (int((w/h)*L),int(L))
        else: # w>h
            img_s = (int(L),int((h/w)*L))
        return cv2.resize(img,dsize=img_s,interpolation=cv2.INTER_CUBIC)
    else:
        print(f'*** WARNING: module cv2 not found, cannot resize figure to fit in size {img_s}')

def get_max_width_height(img_lst):
    if not _is_valid_list_of_img(img_lst):
        raise ValueError('img_lst must be a list or a np.ndarray of objects')
    h_max = max(I.shape[0] for I in img_lst)
    w_max = max(I.shape[1] for I in img_lst)
    return w_max,h_max

def get_image_pattern(img,L=None,bg_state=1,flatten=False, dtype=None):
    if _is_valid_list_of_img(img):
        s = max(get_max_width_height(img))
        return [ get_image_pattern(_resize_img_cv2(I,s),L=L,bg_state=bg_state,flatten=flatten,dtype=dtype) for I in img ]
    if not img.flags['WRITEABLE']:
        img = img.copy()
    if _exists(L):
        img = _resize_img_cv2(img,L)
    h,w         = img.shape[:2]
    s           = max((h,w))
    i0,j0       = (0,max((s//2 - w//2,0))) if h>=w else (max((s//2-h//2,0)),0)
    img         = img.astype(dtype)
    img[img>0]  =  1
    img[img==0] = -1
    I                      = np.full((s,s),bg_state,dtype=dtype)
    I[i0:(i0+h),j0:(j0+w)] = img
    return I.flatten() if flatten else I

def initialize_hopfield_model(patterns):
    """
    Initialize the weight matrix of a Hopfield network using Hebbian learning.

    This function constructs the symmetric weight matrix `W` from a set of 
    reference patterns. Each pattern contributes to the weights via the 
    outer product rule, normalized by the number of neurons. The diagonal 
    elements of `W` are set to zero to avoid self-connections.

    Parameters
    ----------
    patterns : array-like or list of numpy.ndarray
        A collection of reference patterns used to train the Hopfield network.
        Each pattern should be a 1D array of length N (flattened vector). 
        Values are typically ±1.

    Returns
    -------
    W : numpy.ndarray
        The weight matrix of shape (N, N), initialized according to Hebbian learning.
    N : int
        The number of neurons (length of each pattern).
    patterns : list of numpy.ndarray
        The processed list of patterns used for initialization.

    Notes
    -----
    - Hebbian learning rule:
      
          W = Σ (xi ⊗ xi) / N

      where `xi` is a pattern vector and ⊗ denotes the outer product.
    - The diagonal of `W` is set to zero to prevent self-feedback.
    - This initialization allows the Hopfield network to store the given 
      patterns as attractors.

    Examples
    --------
    >>> patterns = [np.array([1, -1, 1]), np.array([-1, -1, 1])]
    >>> W, N, processed_patterns = initialize_hopfield_model(patterns)
    >>> print(N)
    3
    >>> print(W)
    [[ 0.  -0.5  0.5]
     [-0.5  0.   0. ]
     [ 0.5  0.   0. ]]
    """
    patterns = _make_list(patterns)
    N        = len(patterns[0])
    W        = np.zeros((N, N))
    for xi in patterns:
        W += np.outer(xi, xi) / N
    np.fill_diagonal(W, 0)
    return W, N, patterns

def calculate_energy(W, s):
    """
    Compute the energy of a Hopfield network state.

    The energy function is defined as:

        E = -0.5 * s^T W s

    where `s` is the current state vector and `W` is the symmetric weight matrix.
    This energy formulation ensures that the Hopfield network dynamics converge 
    toward stable states (local minima of the energy landscape).

    Parameters
    ----------
    W : numpy.ndarray
        Symmetric weight matrix of shape (N, N), where N is the number of neurons.
    s : numpy.ndarray
        State vector of shape (N,), with entries typically ±1.

    Returns
    -------
    float
        The scalar energy value of the current state.

    Notes
    -----
    - Lower energy values correspond to more stable states.
    - The diagonal of `W` is typically set to zero to avoid self-connections.
    - This energy function is central to analyzing convergence in Hopfield networks.

    Examples
    --------
    >>> W = np.array([[0, 1], [1, 0]])
    >>> s = np.array([1, -1])
    >>> calculate_energy(W, s)
    -1.0
    """
    return -0.5 * s.T @ W @ s

def calculate_overlap(xi, s):
    """
    Compute the overlap between a state vector and one or more reference patterns.

    The overlap is defined as the normalized dot product between the state `s` 
    and each reference pattern `xi`:

        m = (1 / N) * (xi · s)

    where N is the number of neurons (length of `s`).

    Parameters
    ----------
    xi : array-like, list, or numpy.ndarray
        Reference pattern(s) to compare against:
        - If `xi` is a list of vectors, returns a list of overlaps.
        - If `xi` is a single vector of shape (N,), returns a scalar overlap.
        - If `xi` is a matrix of shape (P, N), with one pattern per row, 
          returns a vector of overlaps of length P.
    s : numpy.ndarray
        State vector of shape (N,), with entries typically ±1.

    Returns
    -------
    overlap : float, numpy.ndarray, or list
        - Scalar (float) if `xi` is a single vector.
        - 1D numpy.ndarray of shape (P,) if `xi` is a matrix.
        - List of floats if `xi` is a list of vectors.

    Notes
    -----
    - The overlap measures similarity between the current state and stored patterns.
    - Values close to +1 indicate strong alignment, while values near -1 indicate 
      strong anti-alignment.

    Examples
    --------
    >>> s = np.array([1, -1, 1, -1])
    >>> xi = np.array([1, -1, 1, -1])
    >>> calculate_overlap(xi, s)
    1.0

    >>> xi_matrix = np.array([[1, -1, 1, -1],
    ...                       [-1, 1, -1, 1]])
    >>> calculate_overlap(xi_matrix, s)
    array([ 1., -1.])

    >>> xi_list = [np.array([1, -1, 1, -1]), np.array([-1, 1, -1, 1])]
    >>> calculate_overlap(xi_list, s)
    [1.0, -1.0]
    """
    if type(xi) is list:
        return [ calculate_overlap(xxi,s) for xxi in xi ]
    return (1.0/s.size)*np.dot(xi,s)

def iterate_hopfield_synchronous(W, s_init, max_iter=15, patterns=None, save_energy=True, save_net_state=False):
    """
    Perform synchronous updates in a Hopfield network and track the system's energy and overlap (if patterns is given).

    This function simulates the retrieval dynamics of a Hopfield network using 
    synchronous updates, where all neurons are updated simultaneously according to:

        S(t+1) = sign(W @ S(t))

    The energy of the system at each step is computed as:

        E = -0.5 * S^T * W * S

    If reference patterns are provided, the function also computes the overlap 
    between the evolving state and each stored pattern at every iteration.

    Parameters
    ----------
    W : numpy.ndarray
        Symmetric weight matrix of shape (N, N), where N is the number of neurons.
    s_init : numpy.ndarray
        Initial state vector of shape (N,), with entries typically ±1.
    max_iter : int, optional (default=15)
        Maximum number of synchronous update iterations to perform.
    patterns : array-like or None, optional
        Reference patterns to compare against. If provided, should be an array-like 
        object of shape (P, N), where P is the number of patterns. Overlaps with 
        each pattern are computed at every iteration.
    save_energy : bool
        if True, saves energy at every iteration
    save_net_state : bool
        if True, saves network state for every iteration and returns it 

    Returns
    -------
    s : numpy.ndarray
        Final state vector after convergence or reaching `max_iter`.
    E_data : numpy.ndarray
        Array of energies at each iteration, shape (T,), where T ≤ max_iter.
    m : numpy.ndarray
        Overlap values with each pattern at each iteration, shape (P, T).
        If `patterns` is None, an empty array is returned.
    s_data : numpy.ndarray, shape (N,T)
        s_data[:,t] -> state of all neurons (spins) at time t
        if save_net_state == False, then returns just an empty array

    Notes
    -----
    - The update rule uses `np.sign(W @ s)`. Any zero entries are set to +1.
    - Iteration stops early if the state does not change between successive updates.
    - Overlap is typically defined as the normalized dot product between the 
      current state and each reference pattern.

    Examples
    --------
    >>> W = np.array([[0, 1], [1, 0]])
    >>> s_init = np.array([1, -1])
    >>> patterns = [[1, -1], [-1, 1]]
    >>> s, E, m = iterate_hopfield_synchronous(W, s_init, max_iter=10, patterns=patterns)
    >>> print(s)
    [1 -1]
    >>> print(E)
    [-1.0, -1.0]
    >>> print(m)
    [[1.0, 1.0],
     [-1.0, -1.0]]
    """
    has_patterns   = _exists(patterns)
    N              = len(s_init)
    s0             = s_init.copy().astype(float)
    s_data         = np.empty((N,max_iter if save_net_state else 0),dtype=float)
    if save_net_state:
        s_data[0,:] = s
    if save_energy:
        E_data     = np.empty(max_iter,dtype=float)
        E_data[0]  = calculate_energy(W,s0)    
    else:
        E_data     = np.empty(0,dtype=float)
    m              = np.empty((0,0),dtype=float)
    if has_patterns:
        patterns = np.atleast_2d(patterns) #np.array(_make_list(patterns))
        P        = patterns.shape[0]
        m        = np.empty((P,max_iter),dtype=float)
        m[:,0]   = calculate_overlap(patterns,s0)
    
    for t in range(1,max_iter):
        s         = np.sign(W @ s0)
        s[s == 0] = 1
        
        # Calculate energy of the new state
        if save_energy:
            E_data[t]  = calculate_energy(W,s)
        if has_patterns:
            m[:,t] = calculate_overlap(patterns,s)
        if save_net_state:
            s_data[:,t] = s
        if np.array_equal(s, s0):
            break
        s0 = s
        
    return s, E_data[:(t+1)], m[:,:(t+1)], s_data[:,:(t+1)]

import numpy as np

def _events_to_spins(ds_evt, S0, T):
    """
    ds_evt : list of (t, i, ds)
    S0     : array of shape (N,) with initial spins (+1 / -1)
    T      : final time
    """
    N = len(S0)
    S = np.zeros((N, T+1))

    # initial condition
    S[:, 0] = S0

    # copy forward
    for t in range(1, T+1):
        S[:, t] = S[:, t-1]

    # apply events
    for t, i, ds in ds_evt:
        S[i, t:] += ds   # ds = ±2 flips the spin

    return S

def iterate_hopfield_sequential(W, s_init, max_MCsteps=10, patterns=None, save_energy=True, save_net_state=False):
    """
    Perform asynchronous (sequential) updates in a Hopfield network and track the system's energy.

    This function simulates the retrieval dynamics of a Hopfield network using 
    asynchronous (sequential) updates, where neurons are updated one by one in a 
    random order during each Monte Carlo (MC) step (epoch). After every single 
    neuron update, the energy of the system is recorded. If reference patterns 
    are provided, the overlap between the evolving state and each stored pattern 
    is also tracked.

    Parameters
    ----------
    W : numpy.ndarray
        Symmetric weight matrix of shape (N, N), where N is the number of neurons.
    s_init : numpy.ndarray
        Initial state vector of shape (N,), with entries typically ±1.
    max_MCsteps : int, optional (default=10)
        Maximum number of Monte Carlo steps (epochs). Each step updates all neurons once.
    patterns : array-like or None, optional
        Reference patterns to compare against. If provided, should be an array-like 
        object of shape (P, N), where P is the number of patterns. Overlaps with 
        each pattern are computed after every neuron update.
    save_energy : bool
        if True, saves energy at every iteration
    save_net_state : bool
        if set, saves all activation and deactivation events by saving tuples (t,i,ds), spin i changed by ds at time t
        i.e., ds = ds_i[t] = s_i[t]-s_i[t-1]
        ds = +- 2 for standard sign function (0-temperature Hopfield model; +2 for activation; -2 for deactivation)

    Returns
    -------
    s : numpy.ndarray
        Final state vector after convergence or reaching `max_MCsteps`.
    E_data : numpy.ndarray
        Array of energies recorded after every single neuron update, 
        shape (T,), where T ≤ max_MCsteps * N.
    m : numpy.ndarray
        Overlap values with each pattern at each update, shape (P, T).
        If `patterns` is None, an empty array is returned.
    ds_evt : list of tuple
        ds_evt = [ (t1,i1,ds1), (t2,i2,ds2), ... ]
        where (t,i,ds) are the change ds of spin i at time t
        ds_evt = empty list if save_net_state == False

    Notes
    -----
    - The update rule for neuron i is based on its local field:
      
          h_i = Σ_j W_ij * s_j
          s_i = sign(h_i), with ties resolved as +1

    - Neurons are updated in a random order at each MC step.
    - Iteration stops early if no neurons change state during a full pass, 
      indicating convergence to a local minimum.
    - Overlap is typically defined as the normalized dot product between the 
      current state and each reference pattern.

    Examples
    --------
    >>> W = np.array([[0, 1], [1, 0]])
    >>> s_init = np.array([1, -1])
    >>> patterns = [[1, -1], [-1, 1]]
    >>> s, E, m = iterate_hopfield_sequential(W, s_init, max_MCsteps=5, patterns=patterns)
    >>> print(s)
    [1. -1.]
    >>> print(E[:5])
    [-1.0, -1.0, -1.0, -1.0, -1.0]
    >>> print(m.shape)
    (2, 5)
    """
    N            = len(s_init)
    indices      = np.arange(N)
    has_patterns = _exists(patterns)

    # Record initial energy
    tTotal    = max_MCsteps * N
    s         = s_init.copy().astype(float)
    ds_evt    = []
    #if save_net_state:
    #    for t in range(s_data.shape[0]):
    #        s_data[t,:] = s
    if save_energy:
        E_data    = np.empty(tTotal, dtype=float)
        E_data[0] = calculate_energy(W,s)
    else:
        E_data    = np.empty(0,dtype=float)
    m         = np.empty((0,0),dtype=float)
    if has_patterns:
        patterns = np.atleast_2d(patterns) #np.array(_make_list(patterns))
        P        = patterns.shape[0]
        m        = np.empty((P,tTotal),dtype=float)
        m[:,0]   = calculate_overlap(patterns,s)
    t = 1
    for t_MC in range(1,max_MCsteps):
        # t_MC = 1 MC step = 1 epoch
        # Create a random order for updating each neuron once per MC step (epoch)
        np.random.shuffle(indices)
        
        state_changed = False
        for i in indices:
            # Calculate the local field for neuron i: h_i = sum(W_ij * s_j)
            h_i     = np.dot(W[i, :], s)
            s_i_new = 1.0 if h_i >= 0 else -1.0
            
            if s_i_new != s[i]:
                if save_net_state:
                    ds_evt.append((t,i,s_i_new - s[i]))
                s[i]  = s_i_new
                state_changed = True
            
            # Track energy after every single neuron update
            if save_energy:
                E_data[t]  = calculate_energy(W,s)
            if has_patterns:
                m[:,t] = calculate_overlap(patterns,s)
            t += 1
        
            
        # If no neurons changed state during a full pass, we've hit a local minimum
        if not state_changed:
            break
    if save_net_state:
        if len(ds_evt)>0:
            s_data = _events_to_spins(ds_evt, s_init, tTotal)[:,:t]
        else:
            s_data = np.tile(s_init.reshape((N,1)),(1,tTotal))[:,:t]
    else:
        s_data = np.empty((N,0),dtype=float)
    return s, E_data[:t], m[:,:t], s_data

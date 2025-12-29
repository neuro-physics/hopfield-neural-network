import numpy as np

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

def iterate_hopfield_synchronous(W, s_init, max_iter=15, patterns=None):
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

    Returns
    -------
    s : numpy.ndarray
        Final state vector after convergence or reaching `max_iter`.
    E_data : numpy.ndarray
        Array of energies at each iteration, shape (T,), where T ≤ max_iter.
    m : numpy.ndarray
        Overlap values with each pattern at each iteration, shape (P, T).
        If `patterns` is None, an empty array is returned.

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
    has_patterns   = type(patterns) is not type(None)
    N              = len(s_init)
    E_data         = np.empty(max_iter,dtype=float)
    m              = np.empty((0,0),dtype=float)
    if has_patterns:
        patterns = np.atleast_2d(patterns) #np.array(_make_list(patterns))
        P        = patterns.shape[0]
        m        = np.empty((P,max_iter),dtype=float)
    
    # Calculate initial energy
    s0        = s_init.copy().astype(float)
    E_data[0] = calculate_energy(W,s0)
    
    for t in range(1,max_iter):
        s         = np.sign(W @ s0)
        s[s == 0] = 1
        
        # Calculate energy of the new state
        E_data[t]  = calculate_energy(W,s)
        if has_patterns:
            m[:,t] = calculate_overlap(patterns,s)

        if np.array_equal(s, s0):
            break
        s0 = s
        
    return s, E_data[:(t+1)], m[:,:(t+1)]

def iterate_hopfield_sequential(W, s_init, max_MCsteps=10, patterns=None):
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
    has_patterns = type(patterns) is not type(None)

    # Record initial energy
    s         = s_init.copy().astype(float)
    E_data    = np.empty(max_MCsteps * N, dtype=float)
    E_data[0] = calculate_energy(W,s)
    m         = np.empty((0,0),dtype=float)
    if has_patterns:
        patterns = np.atleast_2d(patterns) #np.array(_make_list(patterns))
        P        = patterns.shape[0]
        m        = np.empty((P,max_MCsteps * N),dtype=float)
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
                s[i]  = s_i_new
                state_changed = True
            
            # Track energy after every single neuron update
            E_data[t]  = calculate_energy(W,s)
            if has_patterns:
                m[:,t] = calculate_overlap(patterns,s)
            t += 1
            
        # If no neurons changed state during a full pass, we've hit a local minimum
        if not state_changed:
            break
            
    return s, E_data[:t], m[:,:t]

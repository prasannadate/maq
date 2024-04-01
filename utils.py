import numpy as np 
import os 
import pickle



# ******************************************************************************
# API Functions ****************************************************************
# ******************************************************************************


def array2Dict(x):
	""" Converts numpy 1-dimensional array into dictionary.

	Args:
		x (numpy array): Input array

	Returns:
		d (dict): Dictionary with array indices as keys and array values as values.

	"""

	d = {}
	for i in range(len(x)):
		if x[i] != 0.0:
			d[(i, i)] = x[i]

	return d




def embedChimeraIsing(Q, p, M=16, N=16, L=4, Lambda=None, normalize=True):
	""" Generates embedding for an Ising problem characterized by Q and p for D-Wave Chimera hardware.

	Args:
		Q (numpy array):
		p (numpy array):
		M (int):
		N (int):
		L (int):

	Returns:
		J (dict):
		h (dict):

	Raises:
		TypeError: If M, N or L are not integers
		ValueError: If Lambda is given and not positive

	"""

	if not isinstance(M, int):
		raise TypeError("'M' must be an integer")
	if not isinstance(N, int):
		raise TypeError("'N' must be an integer")
	if not isinstance(L, int):
		raise TypeError("'L' must be an integer")

	maximum = max(np.absolute(Q).max(), np.absolute(p).max())

	if Lambda:
		if Lambda < 0:
			raise ValueError("'Lambda' must be a large positive integer")
	else:
		Lambda = max(Q.max(), p.max()) * 1000

	Q = Q.copy()
	p = p.copy()

	# Normalize
	if normalize:
		Q = Q / maximum
		p = p / maximum

	Q = _makeUpperTriangular(_zeroDiagonal(Q))

	J, h, embeddings, qubitFootprint = _computeEmbeddings(Q, p, M, N, L, Lambda)

	return J, h, embeddings, qubitFootprint




def embedChimeraQubo(Q, p, M=16, N=16, L=4, Lambda=None, normalize=True):
	""" Generates embedding for an Ising problem characterized by Q and p for D-Wave Chimera hardware.

	Args:
		Q (numpy array):
		p (numpy array):
		M (int):
		N (int):
		L (int):

	Returns:
		J (dict):
		h (dict):

	Raises:
		TypeError: If M, N or L are not integers
		ValueError: If Lambda is given and not positive

	"""

	if not isinstance(M, int):
			raise TypeError("'M' must be an integer")
	if not isinstance(N, int):
		raise TypeError("'N' must be an integer")
	if not isinstance(L, int):
		raise TypeError("'L' must be an integer")

	maximum = max(np.absolute(Q).max(), np.absolute(p).max())

	if Lambda:
		if Lambda < 0:
			raise ValueError("'Lambda' must be a large positive integer")
	else:
		Lambda = maximum * 8.0

	Q = Q.copy()
	p = p.copy()

	# Normalize
	if normalize:
		Q = Q / maximum
		p = p / maximum

	# Transfer diagonal elements of Q to p
	for i in range(Q.shape[0]):
		p[i] += Q[i,i]
		Q[i,i] = 0

	Q = _makeUpperTriangular(Q)

	J, h, embeddings, qubitFootprint = _computeEmbeddings(Q, p, M, N, L, Lambda)

	h = array2Dict(h)
	J.update(h)

	# Correct h for QUBO embeddings
	for qubitSet in embeddings:
		length = len(qubitSet)
		for qubit in qubitSet:
			if (qubit, qubit) in J:
				J[(qubit, qubit)] += (length-1)*Lambda / length
			else:
				J[(qubit, qubit)] = (length-1)*Lambda / length

	return J, embeddings, qubitFootprint




def postProcessing(response, embeddings, A, b=None):
	""" Post-process the response obtained from D-Wave.

	Args:
		response: response object returned by D-Wave Ocean API

	Returns:
		results: List of (solution, energy, number of occurrences) tuples

	"""

	optimalValue = np.inf 
	optimalSolution = []
	optimalSolutionSet = set()

	indices = np.array([e[0] for e in embeddings])
	for sample, _, _ in response.data():
		solution = np.array([sample[i] for i in indices])
		solutionTuple = tuple(solution)

		value = solution.T @ A @ solution + solution.T @ b
		if value < optimalValue:
			optimalValue = value
			optimalSolution = [solution.copy()]
			optimalSolutionSet.add(solutionTuple)
		elif value == optimalValue and solutionTuple not in optimalSolutionSet:
			optimalSolution.append(solution.copy())
			optimalSolutionSet.add(tuple(solution))

	return optimalValue, optimalSolution




# ******************************************************************************
# Helper Functions *************************************************************
# ******************************************************************************


def _baseQubit(M, N, L, V, i):
	""" Compute index of the base qubit corresponding to input variable i.

	Helper function to _compute_embeddings().

	Args:
		M: Number of rows in hardware Graph
		N: Number of columns in hardware Graph
		L: Number of qubits along a line in a single hardware block
		i: Index of input variable being evaluated

	Returns:
		Base qubit corresponding to input variable i.

	"""
	
	return int(2 * L * np.floor(i/L) * (min(M,N) + 1) + i % L)




def _computeCouplings(M, N, L, V):
	""" Compute the couplings.

	Args:
		M: Number of rows of Chimera hardware architecture
		N: Number of columns of Chimera hardware architecture
		L: Number of qubits along a line in a single Chimera hardware block
		V: Number of input variables

	Returns:
		couplings: Dictionary mapping variables (keys) to qubits (values)

	Raises:
		RuntimeError: If cannot accommodate V variables on M X N X L hardware graph

	"""

	couplings = {}

	if V <= min(M, N) * L:
		vStep = _verticalStep(M, N, L)
		hStep = _horizontalStep(L)
		for i in range(int(V)):
			startQubit = _startQubit(i, L)
			baseQubit = _baseQubit(M, N, L, V, i)
			endQubit = _endQubit(baseQubit, L, V, i)

			# Special case: If V % L = 1, and dealing with last qubit, only need "vertical couplings"
			if (V % L == 1) and (i == V - 1):
				couplings[i] = list(range(startQubit, baseQubit, vStep))
			
			# General case: For all other coupled qubits
			else:
				couplings[i] = list(range(startQubit, baseQubit, vStep)) + [baseQubit] + list(range(int(baseQubit + L), endQubit, hStep))

	else:
		raise RuntimeError("Can not embed %d variables on %d X %d X %d hardware graph" %(V, M, N, L))

	return couplings




def _computeEmbeddings(Q, p, M, N, L, Lambda):
	""" Computes the complete embeddings required to embed a QUBO / Ising problem onto hardware graph.
	
	First creates a dictionary of all the qubits that will be coupled to an original variable of the problem.
	Then, sets the inter-qubit coupling strengths as per input matrix Q.
	
	Args:
		Q: QUBO / Ising matrix (should be upper triangular with zeroes on the main diagonal)
		p: QUBO / Ising vector
		M: Number of rows in Chimera hardware graph
		N: Number of columns in Chimera hardware graph
		L: Number of qubits along a line in a single hardware block
		Lambda: Penalty value (large positive number)

	Returns:
		J: Dictionary of inter-qubit connections (keys) along with their coupling strength (values)
		h: Qubit biases
		embeddings: List containing lists of qubits mapped to a particular qubit
		qubitFootprint: Total number of qubits used

	"""

	# Define the variables to be computed
	V = Q.shape[0]
	qubitFootprint = _qubitFootprint(M, N, L, V)
	if V <= L + 1:
		h = np.zeros(V + L - 1)
	else:
		h = np.zeros(int(2*L*np.floor(float(V)/float(L)) + V%L))
	J = {}
	embeddings = []
	numCoupledQubits = 0


	# If problem fits on a single block of Chimera hardware graph
	if V <= L + 1:
		if V == 0:
			return ({}, np.array([]), [], qubitFootprint)
		elif V == 1:
			return ({}, p, [[0]], qubitFootprint)
		else:
			for i in range(V):
				if i == V-1:
					embeddings.append([i+L-1])
				else:
					embeddings.append([i])

				if i == V-1:
					h[i+L-1] = p[i]
				
				else:
					h[i] = p[i]
					if i > 0:
						J[i, i+L-1] = -Lambda
						embeddings[i].append(i+L-1)
						numCoupledQubits += 1
				
				for j in range(i+L, L+V-1):
					J[(i,j)] = Q[i,j-L+1]

			return (J, h, embeddings, qubitFootprint)


	if os.path.exists("./embeddings/%d/%d.pkl" %(M, V)):
		with open("./embeddings/%d/%d.pkl" %(M, V), "rb") as file:
			couplings = pickle.load(file)
	else:
		couplings = _computeCouplings(M, N, L, V)


	# Compute embeddings
	for i in range(V):
		embeddings.append(couplings[i][:])


	# Set the qubit weight vector h
	for i in range(int(V)):
		h[_startQubit(i, L)] = p[i]


	# Set the coupled inter-qubit strengths
	for i in couplings:
		for j in range(1, len(couplings[i])):
			J[(couplings[i][j-1], couplings[i][j])] = -Lambda
			numCoupledQubits += 1


	# Set the non-coupled inter-qubit strengths
	for i in range(0, int(V), int(L)):
		for j in range(i, int(V), int(L)):
			if i == j:
				for s in range(i, int(min(i+L, V))):
					for t in range(s+1, int(min(i+L, V))):
						J[(couplings[s][0], couplings[t][1])] = Q[s,t]

			else:
				for s in range(i, int(min(i+L, V))):
					for t in range(j, int(min(j+L, V))):
						J[(couplings[t][0], couplings[s][0])] = Q[s,t]

			# Pop completed qubits from couplings
			for s in range(i, int(min(i+L, V))):
				if couplings[s]:
					couplings[s].pop(0)
			for t in range(j, int(min(j+L, V))):
				if couplings[t]:
					couplings[t].pop(0)

	return J, h, embeddings, qubitFootprint




def _decode(sample, indices):
	""" Decodes the sample returned by D-Wave

	Args:
		sample: A sample returned by D-Wave
		indices: Indices of base qubits for Ising / QUBO variables

	Returns:
		solution: Decoded solution

	"""

	solution = []
	for index in indices:
		solution.append(sample[index])

	return np.array(solution).astype(np.int64)




def _endQubit(base, L, V, i):
	""" Compute the index of final qubit that would be coupled to input variable i.

	Helper function to _compute_embeddings().

	Args:
		L: Number of qubits along a line in a single hardware block
		V: Number of input variables
		i: Index of input variable being evaluated
	
	Returns:
		Index of the final qubit that would be coupled to input variable i.

	"""

	return int(base + 2 * L * (np.ceil(float(V)/float(L)) - np.floor(i/L)))




def _horizontalStep(L):
	""" Compute number of qubits that need to be skipped for horizontal coupling.

	Helper function to _compute_embeddings().

	Args:
		L: Number of qubits along a line in a single hardware block

	Returns:
		Step size for howizontal coupling.

	"""

	return int(2 * L)




def _makeUpperTriangular(Q):
	""" Makes the matrix Q upper triangular by adding entries from lower triangular part into the upper triangular part.
		
	Args:
		Q: The input matrix.

	Returns:
		Q: The upper triangular matrix Q[i,j] = Q[i,j] + Q[j,i], for all i < j

	"""

	for i in range(Q.shape[0]):
		for j in range(i+1, Q.shape[0]):
			Q[i,j] = Q[i,j] + Q[j,i]
			Q[j,i] = 0

	return Q




def _qubitFootprint(M, N, L, V):
	""" Computes total number of qubits required to embed V QUBO variables onto hardware Graph defined by M, N and L.

	Args:
		M: Number of rows in Chimera hardware graph
		N: Number of columns in Chimera hardware graph
		L: Number of qubits along a line in a single Chimera hardware block
		V: Number of input variables

	Returns:
		Number of qubits required if V is smaller than L * min(M, N), otherwise returns None.

	"""

	# Handle the base case for single hardware block
	if V <= L + 1:
		if V == 0 or V == 1:
			return V
		else:
			return 2 * (V - 1)

	if V <= L * min(M,N):
		if V % L == 0:
			return int(L * np.floor(V/L) * (np.floor(V/L) + 1))
		elif V % L == 1:
			return int(L * np.floor(V/L) * (np.floor(V/L) + 1) + L * np.floor(V/L) + (V % L) * np.floor(V/L))
		else:
			return int(L * np.floor(V/L) * (np.floor(V/L) + 1) + L * np.floor(V/L) + (V % L) * np.floor(V/L) + 2 * (V % L))
	else:
		return None




def _startQubit(i, L):
	""" Compute the smallest hardware Graph index of the qubit that is mapped to given variable i.

	Helper function to _compute_embeddings().

	Args:
		i: Index of input variable being evaluated
		L: Number of qubits along a line in a single hardware block

	Returns:
		Smallest qubit index corresponding to input varible i.

	"""

	return int(2 * L * np.floor(i/L) + i % L)




def _verticalStep(M, N, L):
	""" Compute number of qubits that need to be skipped for vertical coupling.

	Helper function to _compute_embeddings().

	Args:
		M: Number of rows in hardware Graph
		N: Number of columns in hardware Graph
		L: Number of qubits along a line in a single hardware block

	Returns:
		Step size for vertical coupling.

	"""

	return int(2 *  L * min(M, N))




def _zeroDiagonal(Q):
	""" Sets diagonal elements to zero.

		Args:
			Q: The input matrix Q.
		
		Returns:
			Q: Q is the input square matrix whose diagonal has been zeroed out.

	"""

	for i in range(Q.shape[0]):
		Q[i,i] = 0

	return Q







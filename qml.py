import numpy as np




# ******************************************************************************
# API Functions ****************************************************************
# ******************************************************************************


def regression2Qubo(X, Y, P):
	""" Converts a given regression problem into QUBO problem

	Args:
		X (numpy array): Augmented training data having N datapoints and (d+1) features, i.e. d features augmented by unity
		Y (numpy array): Training labels for regression, N dimensional vector
		P (numpy array): Precision vector, must be sorted, can contain positive and negative powers of 2
	
	Returns:
		Q (numpy array): QUBO matrix
		p (numpy array): QUBO vector

	Raises:
		TypeError: If X, Y or P are not numpy arrays
		ValueError: 
			1. If shape of X is not [N,d] 
			2. If shape of Y is not [N]
			3. If first dimensions of X and Y are not equal

	"""

	if not isinstance(X, np.ndarray):
		raise TypeError("X must be numpy array")
	if not isinstance(Y, np.ndarray):
		raise TypeError("Y must be numpy array")
	if not isinstance(P, np.ndarray):
		raise TypeError("P must be numpy array")
	if len(X.shape) != 2:
		raise ValueError("X must be a 2-dimensional (Nxd) numpy array")
	if len(Y.shape) != 1:
		raise ValueError("Y must be a 1-dimensional numpy array")
	if X.shape[0] != Y.shape[0]:
		raise ValueError("First dimensions of X and Y must be equal")


	# Get N, d, K
	N, d = X.shape
	K = len(P)


	# Compute precision matrix
	precisionMatrix = np.einsum("ik,jl", np.eye(d), P.reshape([1,-1])).reshape([d, K*d])


	# Conversion
	Q = precisionMatrix.T @ X.T @ X @ precisionMatrix
	p = -2 * precisionMatrix.T @ X.T @ Y


	return Q, p




def regressionWeights(solutions, P):
	""" Finds regression weights from binarized regression weights

	Args:
		solutions: List of binarized regression weight vector
		P: Precision vector

	Returns:
		w: List of real valued regression weights

	"""

	weights = []
	K = len(P)

	for wHat in solutions:
		WHat = wHat.reshape([-1, K])
		w = WHat @ P
		weights.append(w)

	return weights




def svm2Qubo(X, Y, P):
	""" Converts a given SVM problem into QUBO problem

	Args:
		X (numpy array): Training data having N datapoints and d features
		Y (numpy array): Training labels for binary classification
		P (numpy array): Precision vector, must be sorted, can contain positive and negative powers of 2
	
	Returns:
		Q (numpy array): QUBO matrix
		p (numpy array): QUBO vector

	Raises:
		TypeError: If X, Y or P are not numpy arrays
		ValueError: 
			1. If shape of X is not [N,d] 
			2. If shape of Y is not [N]
			3. If first dimensions of X and Y are not equal

	"""

	if not isinstance(X, np.ndarray):
		raise TypeError("X must be numpy array")
	if not isinstance(Y, np.ndarray):
		raise TypeError("Y must be numpy array")
	if not isinstance(P, np.ndarray):
		raise TypeError("P must be numpy array")
	if len(X.shape) != 2:
		raise ValueError("X must be a 2-dimensional (Nxd) numpy array")
	if len(Y.shape) != 1:
		raise ValueError("Y must be a 1-dimensional numpy array")
	if X.shape[0] != Y.shape[0]:
		raise ValueError("First dimensions of X and Y must be equal")


	# Get N, d, K
	N, d = X.shape
	K = len(P)


	# Find index of smallest positive element in P
	kplus = 0
	for p in P:
		if p > 0:
			break
		kplus += 1
	Pplus = P[kplus:]
	Kplus = K - kplus


	# Compute outer product of P and Pplus
	outerPlus = np.outer(P, Pplus)


	# Computing QUBO matrix Q
	Q = np.zeros([K*(d+1)+Kplus*N, K*(d+1)+Kplus*N])
	Q[:K*d, :K*d] = - np.einsum("ik,jl", np.eye(d), np.outer(P, P)).reshape((K*d, K*d))
	Q[:K*d, K*(d+1):] = np.einsum("ik,jl", np.multiply(np.tile(Y, (d, 1)), X.T), outerPlus).reshape((K*d, Kplus*N))
	Q[K*d:K*(d+1), K*(d+1):] = np.einsum("ik,jl", Y.reshape((1,-1)), outerPlus).reshape((K, Kplus*N))


	# Computing QUBO vector p
	p = np.hstack((np.zeros(K*(d+1)), - np.ones(N*Kplus)))

	return Q, p




def svmWeights(solutions, P, N, d):
	""" Finds SVM weights from binarized SVM weights

	Args:
		solutions: List of binarized SVM solutions (weights, biases and Lagrangian multipliers)
		P: Precision vector
		N: Size of training dataset
		d: Number of features in training dataset

	Returns:
		weightsList: List of real valued SVM weights
		biasList: List of real valued SVM bias
		lambdas: List of real valued SVM Lagrangian multipliers

	"""

	# Initialize weights, biases and lambdas
	weightsList = []
	biasList = []
	lambdasList = []


	# Find Pplus, Kplus and kplus
	K = len(P)
	kplus = 0
	for p in P:
		if p > 0:
			break
		kplus += 1
	Pplus = P[kplus:]
	Kplus = K - kplus 


	# For each solution, find the weights, biases and lambdas
	for solution in solutions:
		weightsAndBias = solution[:K*(d+1)].reshape([d+1, K]) @ P
		lambdas = solution[K*(d+1):].reshape([N, Kplus]) @ Pplus
		weightsList.append(weightsAndBias[:-1])
		biasList.append(weightsAndBias[-1])
		lambdasList.append(lambdas)


	return weightsList, biasList, lambdasList




def kmeans2Qubo(X, k, alpha=None, beta=None):
	""" Converts a given k-means clustering problem into a QUBO problem

	Args:
		X (numpy array): Data matrix with N rows and d columns
		k (numpy array): Number of clusters desired

	Returns:
		Q (numpy array): QUBO matrix
		p (numpy array): QUBO vector

	Raises:
		TypeError: 
			1. If X is not numpy array
			2. If k is not an integer
		ValueError: If shape of X is not [N, d]

	"""

	if not isinstance(X, np.ndarray):
		raise TypeError("X must be numpy array")
	if not isinstance(k, int):
		raise TypeError("k must be an integer")
	if len(X.shape) != 2:
		raise ValueError("X must be a 2-dimensional (Nxd) numpy array")


	# Define N and d
	N = X.shape[0]
	d = X.shape[1]


	# Create distance matrix
	D = np.zeros([N, N])

	for i in range(N):
		for j in range(N):
			diff = X[i] - X[j]
			D[i,j] = np.dot(diff, diff)


	# Define alpha and beta
	if not alpha:
		alpha = max(D) / (2*(N/k) - 1)

	if not beta:
		beta = max(D)


	# Create QUBO matrix and vector
	Q = np.zeros([N*k, N*k])
	p = np.zeros(N*k)

	for i in range(N):
		for j in range(N):
			for m in range(k):
				Q[N*i + m, N*j + m] += D[i,j] - beta


	for i in range(N):
		for m in range(k):
			for n in range(k):
				Q[N*i + m, N*i + n] += aplpa

	for i in range(N):
		for m in range(k):
			p[N*i + m] += 2* (alpha + N*beta/k)


	return Q, p




def kmeansClusters(solutions, N, k):
	""" Returns the clustering assignment for a given k-means problem

	Args:
		solutions (list): List of k-means solutions returned by the adiabatic quantum computer
		N (int): Number of data points in the original problem
		k (int): Number of clusters in the original problem

	Returns:
		assignmentsList: A list of assignments corresponding to each solution returned by the adiabatic quantum computer

	"""

	# Create the assignments list
	assignmentsList = []


	# Extract assignments from solutions
	for solution in solutions:
		assigment = solution.reshape(N, k)
		assignmentsList.append(assignment)

	return assignmentsList






























#!/usr/bin/python3

import numpy
import scipy
import matplotlib
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from sklearn import datasets
import math
import random

def example():
	iris = datasets.load_iris()

	print('Features: ', iris.feature_names)
	print('Targets: ', iris.target_names)

	petal_lenght = iris.data[:, iris.feature_names.index('petal length (cm)')]
	petal_width = iris.data[:, iris.feature_names.index('petal width (cm)')]

	for target in set(iris.target):
		example_ids = target == iris.target
		print(target, example_ids.shape, petal_lenght[example_ids])
		scatter(petal_lenght[example_ids],
				petal_width[example_ids],
				label = iris.target_names[target],
				color = 'bgr'[target],
				marker = 'x',
				alpha = 0.7)

	unknown = numpy.array([ [1.5, 0.3], 
							[4.5, 1.2],
							[5.5, 2.3],
							[5.1, 1.7] ])

	scatter(unknown[:,0], unknown[:,1], marker = 'v', color = 'gray', s = 50, label = '??')
	xlabel('petal length (cm)')
	ylabel('petal width (cm)')
	grid(True)
	legend(loc = 'upper left')

	show()

def problem2b():
	iris = datasets.load_iris()

	print(matplotlib.rcParams)


	# plt.xkcd()
	matplotlib.rc('font', size = 9)
	# matplotlib.rc('font', family = 'Humor Sans')
	# matplotlib.rc('xtick.major', width = 1.5)
	# matplotlib.rc('ytick.major', width = 1.5)

	L = [None, None, None]
	K = 4

	for y in range(K):
		for x in range(K):
			plt.subplot(K, K, y * K + x +  1)

			# plt.tick_params(axis='x',
							# which='both',
							# top='off')

			#plt.tick_params(axis='y',
							# which='both',
							# right='off')

			if x == 0:
				plt.ylabel(iris.feature_names[y])

			if y == K - 1:
				plt.xlabel(iris.feature_names[x])

			if x != y:
				for target in set(iris.target):
					example_ids = target == iris.target
					data = scatter(iris.data[:, x][example_ids], 
							iris.data[:, y][example_ids],
							color = 'bgr'[target],
							marker = 'o')

					L[target] = data
			else:
				hist(iris.data[:, x])

	plt.figlegend(L, 
				  [iris.target_names[i] for i in range(3)],
				  numpoints=3, 
				  loc = 'upper center',
				  ncol = 3)

	show()

def kNN(training_targets, training_data, unknown_data, k):
	result = []

	for x in unknown_data:
		dists_sq = numpy.array(list(map(lambda o: numpy.dot(o, o), training_data - x)))
		perm = numpy.argsort(dists_sq)

		# print(dists_sq)

		counts = {}

		for i in range(k):
			# dist = math.sqrt(dists_sq[perm[i]])
			# dist_inv = float("+inf") if dist == 0.0 else 1 / dist

			if training_targets[perm[i]] in counts:
				counts[training_targets[perm[i]]] += 1
			else:
				counts[training_targets[perm[i]]] = 1

		T = sorted(list(counts.items()), key = lambda x: -x[1])

		if len(T) > 1 and T[0][1] == T[1][1]:
			result.append(T[random.randint(0, 1)][0])
		else:
			result.append(T[0][0])
		
		# result.append(max(counts.items(), key = lambda x: x[1])[0])

	return result

def problem4():
	iris = datasets.load_iris()

	def run_once(k):
		perm = numpy.random.permutation(len(iris.target))

		perm_target = numpy.empty_like(iris.target)
		perm_data = numpy.empty_like(iris.data)

		for i, x in enumerate(perm):
			perm_target[i] = iris.target[perm[i]]
			perm_data[i] = iris.data[perm[i]]

		training_set_size = math.floor(len(perm) * 0.666)

		result = kNN(perm_target[:training_set_size], 
					 perm_data[:training_set_size],
					 perm_data[training_set_size:],
					 k)

		num_errors = 0

		for i in range(training_set_size, len(perm)):
			if result[i - training_set_size] != perm_target[i]:
				num_errors += 1

		return num_errors / len(result)

	def run_many(n, k):
		sum = 0
		for i in range(n):
			sum += run_once(k)

		return sum / n

	X = list(range(1, 20, 2))
	Y = []

	for i in range(1, 20, 2):
		Y.append(run_many(500, i))
		print(i, ": ", Y[-1])

	plt.plot(X, Y, '-x')
	plt.xlabel("k")
	plt.ylabel("Average classification error [%]")
	plt.title("Iris classification using k-NN")
	plt.show()

def problem9(n, p):
	samples = numpy.random.binomial(n, p, n)

	def binomial(x, n, p):
		return scipy.special.binom(n, x) * p ** x * (1 - p) ** (n - x)

	def L(p):
		X = numpy.empty(len(samples), dtype = numpy.float64)

		for i, y in enumerate(map(lambda x: binomial(x, n, p), samples)):
			X[i] = y

		return numpy.prod(X)

	X = numpy.linspace(0, 1.0, 1000)
	Y = numpy.empty(len(X), dtype = numpy.float64)

	for i, x in enumerate(X):
		Y[i] = math.pow(10, L(x))

	plt.plot(X, Y)
	plt.axvline(x = numpy.mean(samples) / n, ymin = 0.0, ymax = 1.0, linewidth=2, color='r')
	# plt.ylim(-2000, 0)
	plt.show()


# example()
# problem2b()
# problem4()

problem9(10, 0.4)

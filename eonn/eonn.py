# Copyright (C) 2009 Rogier Koppejan <rogier.koppejan@gmail.com>
#
# This file is part of eonn.
#
# Eonn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Eonn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this library.	If not, see <http://www.gnu.org/licenses/>.

"""
Main module, implementing a complete generational evolutionary algorithm.
"""

import random
from organism import *
import numpy as np
from sklearn.gaussian_process import *
import random
from pprint import *

samplesize	= 5			# Sample size used for tournament selection
keep		= 0			# Nr. of organisms copied to the next generation (elitism)
mutate_prob = 0.75	# Probability that offspring is being mutated
mutate_frac = 0.2		# Fraction of genes that get mutated
mutate_std	= 1.0		# Std. dev. of mutation distribution (gaussian)
mutate_repl = 0.25	# Probability that a gene gets replaced


def optimize(pool, feval, epochs=100, verbose=False):
	""" Evolve supplied population using feval as fitness function.

	Keyword arguments:
	pool	 -- population to optimize
	feval	 -- fitness function
	epochs -- duration of evolutionary run

	"""
	
	# Matrx X used in Gaussian Fit: sizePool x nrVars+2 (for Z)
	X = np.empty([len(pool),len(pool[0].genome)+2])
	for n,org in enumerate(pool):
		for i,gene in enumerate(org.genome):
			X[n][i] = gene.dna[-1]
	y = np.empty(len(pool))

	# Create returns in combination with z for original data
	for i,org in enumerate(pool):
		z = np.random.uniform(0,1,2)
		reward = feval(org.policy,list(z))
		org.evals.append(reward)
		y[i] = reward
		X[i][-2] = z[0]
		X[i][-1] = z[1]
	
	# do GP fit and a evaluation for each epoch
	for i in xrange(epochs):
		print i

		# pool pi used for exploring landscape
		pool = epoch(pool, 4 * len(pool))
		# the gp fit
		gp = GaussianProcess()
		gp.fit(X,y)

		# pool z used for exploring landscape (grid)
		zPool = np.linspace(0,1,10)

		# create pi z pairs (from pool) to investigate landscape
		x = []
		for org in pool:
			for z1 in zPool:
				for z2 in zPool:
					g = getWeights(org)
					g.append(z1)
					g.append(z2)
					x.append(g)

		# evaluate gridsearch
		x = np.array(x)
		yPred, MSE = gp.predict(x, eval_MSE=True) # actual landscape

		UCB = yPred + 1.96 * np.sqrt(MSE) # implement here for smarter search through landscape

		# select interesting pi-z pairs
		sortedUCB = np.argsort(UCB) # dito above
		#bestPiZ = x[sortedUCB][-5:]
		#bestPiZ = acquisitionFunction(pool, zPool, UCB);
		bestPiZ = acquisitionFunction_2(pool, zPool, UCB, yPred, x)

		# holds the pi evaluated in this epoch and will be used in the next epoch
		orgList = []

		# evaluate selecte pi-z pairs
		for piZ in bestPiZ:
			pi = piZ[:-2]
			org = pool.find(pi)
			orgList.append(org)
			
			# if already evaluated, skip???
			if piZ.tolist() in X.tolist():
				continue
			
			z = piZ[-2:]
			reward = feval(org.policy,list(z)) # pi-z pair is evaluated
			org.evals.append(reward)
			piZ = np.atleast_2d(piZ)
			X = np.append(X,piZ,axis = 0)
			y = np.append(y,reward)
		pool = Pool(orgList)
	return pool
			
def epoch(pool, size):
	""" Breed a new generation of organisms."""
	offspring = []
	elite = sorted(pool, reverse=True)[:keep]
	for i in range(size - keep):
		mom, dad = select(pool), select(pool)
		offspring.append(reproduce(mom, dad))
	return Pool(offspring + elite)

def reproduce(mom, dad):
	""" Recombine two parents into one organism. """
	child = mom.crossover(dad)
	if random.random() < mutate_prob:
		child.mutate(mutate_frac, mutate_std, mutate_repl)
	return child

def select(pool):
	""" Select one individual using tournament selection. """
	return max(random.sample(pool, samplesize))

def acquisitionFunction(pool, zPool, UCB):
	""" 
	Returns best pi's on the single most interesting
	z. As this function purely focusses on 
	high variation, it is expected to focus on
	keeping alive in annoying situation,
	thus don't expet too much!

	Assumes: a particular ordering in UCB:

	for org in pool
		for z1 in zpool
			for z2 in zpool

	It uses this ordering to reshape the UCB
	and extract variance per z.

	Assumes: UCB is formed by testing out all
	possible pi in policy and z combinations in z
	as indicate above
	"""

	zAmount = len(zPool)
	pAmount = len(pool)

	
	UCBGrid = np.reshape(UCB, (pAmount, zAmount, zAmount))

	# stores variance of UCB of policies on each [z1,z2]
	vars = np.zeros((zAmount, zAmount))

	for (i, j), _ in np.ndenumerate(vars):
		vars[i][j] = np.var( UCBGrid[:][i][j] );

	# return the indices of z with highest variance , argmax provides raveled (flattened)
	varZ = np.unravel_index( np.argmax(vars), vars.shape)

	# sort the policy UCB on that z and return index of top 5 policies
	iBestPi = np.argsort( UCBGrid[:][varZ[0]][varZ[1]] )[-5:] 

	# get weights of each policy and append the z to it
	bestPiZ = []
	for iPi in iBestPi:
		w = getWeights(pool[iPi])
		w.extend(varZ)
		bestPiZ.append(w)
	

	# return as np array
	return np.array(bestPiZ)

def acquisitionFunction_2(pool, zPool, UCB, means, GP):
	""" 
	Returns best 5 policies with UCB, and adds z with highest variances over these policies. 

	Assumes: a particular ordering in UCBfor reshape:

	for org in pool
		for z1 in zpool
			for z2 in zpool
	"""

	zAmount = len(zPool)
	pAmount = len(pool)

	# UCBGrid[1][2][3] = UCB of policy 1 on 
	# [zpool[2], zpool[3]
	UCBGrid = np.reshape(UCB, (pAmount, zAmount, zAmount))
	sortedUCB = np.argsort(UCB) # highest UCB for selecting policies (NB can still be multiple same policies)
	bestPi = GP[sortedUCB][-5:]

	vars = np.zeros((zAmount, zAmount))
	meansGrid = np.reshape(means, (pAmount, zAmount, zAmount))

	policyIndex = np.unique(sortedUCB[-5:]//(zAmount**2)).tolist()

	#calculate variance over selected polocies for each z
	for (i, j), _ in np.ndenumerate(vars):
		vars[i][j] = np.var(meansGrid[[1,2],i,j]);
	# select Z with highest variance
	bestZ = np.unravel_index( np.argmax(vars), vars.shape)	

	# create Pi/Z combinations
	bestPiZ = []
	for pi in bestPi:
		pi.tolist().extend(bestZ)
		bestPiZ.append(pi)
	return np.array(bestPiZ)

def getWeights(organisme):
	""" Returns the weights of an organism """
	w = [0]*len(organisme.genome)
	for i,gene in enumerate(organisme.genome):
		w[i] = gene.dna[-1]

	return w


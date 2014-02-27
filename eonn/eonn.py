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
keep				= 0			# Nr. of organisms copied to the next generation (elitism)
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
	
	# do GP fit and a evaluation
	for i in xrange(epochs):
		# pool pi used for exploring landscape
		pool = epoch(pool, 4 * len(pool))
		# the gp fit
		gp = GaussianProcess()
		gp.fit(X,y)
		# pool z used for exploring landscape
		zPool = np.linspace(0,1,10)
		# create pi z pairs (from pool) to investigate landscape
		x = []
		for org in pool:
			for z1 in zPool:
				for z2 in zPool:
					g = [0]*len(pool[0].genome)
					for i,gene in enumerate(org.genome):
						g[i] = gene.dna[-1]
					g.append(z1)
					g.append(z2)
					x.append(g)
		x = np.array(x)
		yPred, MSE = gp.predict(x, eval_MSE=True) # actual landscape
		UCB = yPred + 1.96 * np.sqrt(MSE) # implement here for smarter search through landscape
		sortedUCB = np.argsort(UCB) # dito above
		bestPiZ = x[sortedUCB][-5:]
		#bestPiZ = acquisiteFunction(pool, zPool, UCB);
		orgList = []
		# extract good pi and z from search
		for piZ in bestPiZ:
			pi = piZ[:-2]
			org = pool.find(pi)
			orgList.append(org)
			z = piZ[-2:]
			reward = feval(org.policy,list(z)) # elite pi z pair is evaluated
			org.evals.append(reward)
			np.append(X,piZ)
			np.append(y,reward)
		pool = Pool(orgList)
	print bestPiZ[:,-2:]
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

def acquisiteFunction(pool, zPool, UCB):
	""" 
		Returns the most interesting piZ pairs 
		As this function almost purely focusses on 
		high variation, it is expected to focus on
		keeping alive in annoying situation,
		thus don't expet too much!
	"""
	
	zAmount = np.power(len(zPool), 2)
	pAmount = len(pool)
	
	# holds the (co)variance of each z 
	vars = np.zeros(zAmount);
	
	# fill vars, where each var has pAmount amount of values
	for i in range(0,zAmount):
		vars[i] = np.var( UCB[ range( (i * pAmount), (i * pAmount) + pAmount   ) ])
	
	# z with highest variance in ucb, where the z is as follows:
	# first element is highest index / len(zPool)
	# second element is highest index mod len(zPool)
	varZ = np.argmax(vars) / len(zPool), np.mod( np.argmax(vars), len(zPool))

	# get 5 best pi on that z
	bestPi = np.argsort(UCB[ range( np.argmax(vars) * pAmount, (np.argmax(vars) + 1) * pAmount ) ])[-5:]
	
	# create 5 best piz from these findings
	bestPiZ = []
	#for pi in bestPi:
		# append them TODO
	
	print "We still have to append the arrays such that bestPiZ is created"
		
	# remove if ^ fixed
	exit()

	# debugging
	print varZ
	print bestPi
	print bestPiZ


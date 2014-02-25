import sys
from eonn import eonn
from eonn.genome import Genome
from eonn.organism import Pool
from math import cos, log, sqrt
import numpy as np
from matplotlib import pyplot as plt
import random as rand

from string import letters, digits
import os

from mpltools import style
style.use('ggplot')

# State limits
LEFT = 0
RIGHT = 1

GOAL = np.array([0.85,0.15])
GOALRADIUS = 0.03

WINDCHANCE = 0.01
WINDSTRENGTH = [0,-0.2]

EPOCHS = 100
ROUNDS = 10

def wind():
	return rand.random() < WINDCHANCE

def update(pos, action):
	""" Updates position with given action. """
	if wind():
		pos += np.array(WINDSTRENGTH)
	return pos + (action * 0.05) 

def checkBounds(pos):
	return (pos < RIGHT).all() and (pos > LEFT).all()
	
def checkGoal(pos):
	dist = np.linalg.norm(pos-GOAL).sum()
	return dist < GOALRADIUS

def draw(l):
	x = [s[0] for s in l]
	y = [s[1] for s in l]
	plt.scatter(x,y)
	plt.xlim([0,1])
	plt.ylim([0,1])
	

def cliff(policy, z = None, max_steps=500, verbose = False):
	""" Cliff evaluation function. """
	if not z:
		z = np.random.uniform(0,1,2)
	pos = z
	l = [pos]
	for i in range(max_steps):
		action = policy.propagate(list(pos),t=1)
		pos = update(pos, np.array(action))
		if verbose:
			l.append(list(pos))
		if checkGoal(pos):
			ret = 0.9 ** i * 1000
			break
		if not checkBounds(pos):
			ret = -1000
			break;
		ret = i
	if verbose:
		draw(l)
	return np.random.uniform(0,1000)


def main():
	""" Main function. """
	pool = Pool.spawn(Genome.open('cliff.net'), 5, std=1)
	
	# Set evolutionary parameters
	eonn.samplesize	 = 5			# Sample size used for tournament selection
	eonn.keep				 = 5			# Nr. of organisms copied to the next generation (elitism)
	eonn.mutate_prob = 0.75		# Probability that offspring is being mutated
	eonn.mutate_frac = 0.2		# Fraction of genes that get mutated
	eonn.mutate_std	 = 0.1		# Std. dev. of mutation distribution (gaussian)
	eonn.mutate_repl = 0.25		# Probability that a gene gets replaced
	
	
	# Evolve population
	pool = eonn.optimize(pool,cliff, epochs = EPOCHS)


if __name__ == '__main__':
	main()


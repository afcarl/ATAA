import sys
from eonn import eonn
from eonn.genome import Genome
from eonn.organism import Pool
from math import cos, log
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import random as rand

from string import letters, digits
import os

# State limits
XMIN =	0
XMAX =	1
YMIN =	0
YMAX =	1

XGOAL = 0.85
YGOAL = 0.15
GOALRADIUS = 0.03

WINDCHANCE = 0.01

EPOCHS = 10
EVALS = 5
ROUNDS = 50

def wind():
	return rand.random() < WINDCHANCE

def update(pos, action):
	""" Updates position and velocity with given action. """
	if(wind()):
		pos += np.array([0,0.1])
	return pos + (action * 0.2) 

def checkBounds(pos):
	return pos[0] < XMAX and pos[0] > XMIN and pos[1] < YMAX and pos[1] > YMIN
	
def checkGoal(pos):
	x = pos[0] < XGOAL + GOALRADIUS and pos[0] > XGOAL - GOALRADIUS
	y = pos[1] < YGOAL + GOALRADIUS and pos[1] > YGOAL - GOALRADIUS
	return x and y

def draw(l):
	x = [s[0] for s in l]
	y = [s[1] for s in l]
	plt.scatter(x,y)
	plt.xlim([0,1])
	plt.ylim([0,1])
	

def cliff(policy, max_steps=500, verbose = False):
	""" Cliff evaluation function. """
	ret = 0
	pos = np.random.rand(1,2)[0]
	l = [pos]
	for i in range(max_steps):
		action = policy.propagate(list(pos),t=1)
		pos = update(pos, np.array(action))
		ret -= 0
		if verbose:
			l.append(list(pos))
		if checkGoal(pos):
			ret += 1000
			break
		if not checkBounds(pos):
			ret -= 1000
			break;
	if verbose:
		draw(l)
	return ret


def main():
	""" Main function. """
	pool = Pool.spawn(Genome.open('cliff.net'), 20, std=1)
	
	# Set evolutionary parameters
	eonn.samplesize	 = 5			# Sample size used for tournament selection
	eonn.keep				 = 5			# Nr. of organisms copied to the next generation (elitism)
	eonn.mutate_prob = 0.75		# Probability that offspring is being mutated
	eonn.mutate_frac = 0.2		# Fraction of genes that get mutated
	eonn.mutate_std	 = 0.1		# Std. dev. of mutation distribution (gaussian)
	eonn.mutate_repl = 0.25		# Probability that a gene gets replaced
	
	
	directory = "pics/"+''.join(rand.sample(letters+digits, 5))
	os.makedirs(directory)
	# Evolve population
	for j in xrange(1,ROUNDS+1):
		pool = eonn.optimize(pool, cliff, epochs=EPOCHS, evals = EVALS)
		print "AFTER EPOCH", j*EPOCHS
		print "average fitness %.1f" % pool.fitness
		champion = max(pool)
		print "champion fitness %.1f" % champion.fitness
		for i in xrange(10):
			cliff(champion.policy,verbose = True)
		plt.savefig(directory+"/"+str(j*EPOCHS)+".png")
		plt.clf()
	with open(directory+'/best.net', 'w') as f:
		f.write('%s' % champion.genome)
	print "Done, everything saved in ", directory
	


if __name__ == '__main__':
	main()


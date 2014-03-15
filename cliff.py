import sys
from eonn import eonn
from eonn.genome import Genome
from eonn.organism import Pool
from eonn.network import Network
from math import cos, log
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import random as rand

from string import letters, digits
import os

# load paths
max_steps = 500

noWindPath = "pure-evolution result/noWind"
littleWindPath = "pure-evolution result/littleWind"
muchWindPath = "pure-evolution result/muchWind"

# save paths
gridPath = "gridImages"

# State limits
XMIN =	0
XMAX =	1
YMIN =	0
YMAX =	1

XGOAL = 0.85
YGOAL = 0.15
GOALRADIUS = 0.03

WINDCHANCE = 0.000
WINDSTRENGTH = 0.2

EPOCHS = 10
EVALS = 5

def wind():
	return rand.random() < WINDCHANCE

def update(pos, action):
	""" Updates position and velocity with given action. """
	if(wind()):
		pos += np.array([0,WINDSTRENGTH])
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
	rounds = 20
	for j in xrange(1,rounds+1):
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
	

def analysis():
	policies = readPolicies()
	
	policyTypes = ["noWind", "littleWind", "muchWind"];
	
	for i in xrange(len(policyTypes)):
		data = analysePolicies(policies[i], policyTypes[i])
		with open(gridPath + "/" +  policyTypes[i] + "/data.txt", 'w') as f:
			f.write(policyTypes[i] + str(data))
	
	print "Done!"
 
def analysePolicies(policies, policy_name):
	
	# data to return
	nrDeaths = 0
	nrSuccess = 0
	averageRet = 0
	
	# create grid
	grid = []
	for i in xrange(1, 10):
		for j in xrange(1, 10):
			grid.append([i, j])
	
	grid = np.array(grid)
	grid = grid / float(10)
	
	# run and plot on grid
	count = 0
	for policy in policies:
		ret = 0
		l = []
		for pos in grid:
	
			# standard episode
			for i in range(max_steps):
				action = policy.propagate(list(pos),t=1)
				pos = update(pos, np.array(action))
				l.append(list(pos))
	
				if checkGoal(pos):
					nrSuccess += 1
					ret += 1000
					break
	
				if not checkBounds(pos):
					nrDeaths += 1
					ret -= 1000
					break;
		# plot run and save
		draw(l)
		plt.savefig(gridPath +  "/" + policy_name + "/" + str(count) + ".png")
		plt.clf()
		
		# add count and return
		count += 1
		averageRet += ret
	
	averageRet /= float(len(policies) * len(grid))
	return "Average return: " + str(averageRet) + "\nDeath count: " + str(nrDeaths) + "\nSucces count: " + str(nrSuccess) 

def readPolicies():
	"""
		Reads in policies from relative
		path littleWindPath and muchWindPath
	"""
	policies = []
	for ptype in [noWindPath, littleWindPath, muchWindPath]:
	
		typePolicies = []
		for policy in os.walk(ptype).next()[1]:
			org = Network(Genome.open(ptype + "/" + policy + "/best.net"))
	
			typePolicies.append(org)
		policies.append(typePolicies)
	
	return policies

if __name__ == '__main__':
	 for i in xrange(10):
		 main()
	#analysis()


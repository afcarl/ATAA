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

regularPolicies = "newProb/regularPolicies"
rarePolicies = "newProb/rarePolicies"
GPSmartPolicies = "newProb/GPSmartPolicies"
GPExtremePolicies = "newProb/GPExtremePolicies"

# save paths
analysisPath = "newProbAnalysis"

# State limits
LEFT = 0
RIGHT = 1

GOAL = np.array([0.85,0.15])
GOALRADIUS = 0.15

WINDSTRENGTH = [0,0]
 
NN_STRUCTURE_FILE = 'noHidden.net'

EPOCHS = 20
EVALS = 5
ROUNDS = 10

def update(pos, action):
	""" Updates position with given action. """
	return pos + (action * 0.05) 

def gust_of_wind(pos):
	return pos + np.array(WINDSTRENGTH)

def checkBounds(pos):
	return (pos < RIGHT).all() and (pos > LEFT).all()
	
def checkGoal(pos):
	dist = np.linalg.norm(pos-GOAL).sum()
	return dist < GOALRADIUS

def draw(l):
	x = [s[0] for s in l]
	y = [s[1] for s in l]
	plt.scatter(x,y, marker = 'x')
	plt.plot([GOAL[0]],[GOAL[1]], 'ro')
	plt.xlim([0,1])
	plt.ylim([0,1])
	

def cliff(genome, z = None, max_steps=500, verbose = False):
	""" Cliff evaluation function. """
	no_wind_yet = False
	policy = genome
	if z == None:
		if np.random.uniform(0, 1) > 0.5:
			z = [0.5]
		else:
			z = [0]

	WINDSTRENGTH[1] = -z[0]
	pos = [0.1,0.1]
	l = [pos]
	ret = 0
	for i in range(max_steps+1):
		action = policy.propagate(list(pos),t=1)
		pos = update(pos, np.array(action))
		l.append(list(pos))
		if checkGoal(pos):
			ret = 0.99 ** i * 100
			break
		if not no_wind_yet and pos[0] > 0.5:
			pos = gust_of_wind(pos)
			no_wind_yet = True
		if not checkBounds(pos):
			break;
		ret = 0
	
	if verbose:
		draw(l)
	return ret
	

def main():
	""" Main function. """
	pool = Pool.spawn(Genome.open(NN_STRUCTURE_FILE), 20, std=1)
	
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

def analysis():
	policies = readPolicies()
	
	policyTypes = ["regularPolicies", "rarePolicies", "GPSmartPolicies", "GPExtremePolicies"];
	
	for i in xrange(len(policyTypes)):
		data = analysePolicies(policies[i], policyTypes[i])
		with open(analysisPath + "/" +  policyTypes[i] + "/data.txt", 'w') as f:
			f.write(policyTypes[i] + str(data))
	
	print "Done!"
 
def analysePolicies(policies, policy_name):
	
	# data to return
	nrDeaths = 0
	nrSuccess = 0
	averageRet = 0
	

	# run and plot on grid
	count = 0
	for policy in policies:

		ret = 0

		no_wind_yet = False
		z = [np.random.uniform(0,0.5)]
		WINDSTRENGTH[1] = -z[0]
		pos = [0.1,0.1]
		l = [pos]
		ret = 0
		for i in range(max_steps+1):
			action = policy.propagate(list(pos),t=1)
			pos = update(pos, np.array(action))
			l.append(list(pos))
			if checkGoal(pos):
				ret = 0.99 ** i * 100
				nrSuccess += 1
				break
			if not no_wind_yet and pos[0] > 0.5:
				pos = gust_of_wind(pos)
				no_wind_yet = True
			if not checkBounds(pos):
				nrDeaths += 1
				break;
		
		# plot run and save
		draw(l)
		plt.savefig(analysisPath +  "/" + policy_name + "/" + str(count) + ".png")
		plt.clf()
		
		# add count and return
		count += 1
		averageRet += ret
	
	averageRet /= float(len(policies))
	return "Average return: " + str(averageRet) + "\nDeath count: " + str(nrDeaths) + "\nSucces count: " + str(nrSuccess) 

def readPolicies():
	"""
		Reads in policies from relative
		path littleWindPath and muchWindPath
	"""
	policies = []
	for ptype in [regularPolicies, rarePolicies, GPSmartPolicies, GPExtremePolicies]:
	
		typePolicies = []
		for policy in os.walk(ptype).next()[1]:
			org = Network(Genome.open(ptype + "/" + policy + "/best.net"))
	
			typePolicies.append(org)
		policies.append(typePolicies)
	
	return policies

if __name__ == '__main__':
	 #for i in xrange(10):
		 #main()
	analysis()


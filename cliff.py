from __future__ import division

from eonn import eonn
from eonn.genome import Genome, BasicGenome
from eonn.organism import Pool, Organism
from eonn.network import Network
import numpy as np
from matplotlib import pyplot as plt
import random as rand
import itertools
import math

from sklearn.gaussian_process import GaussianProcess

try:
	from mpltools import style
	style.use('ggplot')
except:
	pass

# State limits
LEFT = 0
RIGHT = 1

GOAL = np.array([0.85,0.15])
GOALRADIUS = 0.15

WINDCHANCE = 0.01
WINDSTRENGTH = [0,-0.2]
 
NN_STRUCTURE_FILE = 'noHidden.net'

def wind():
	return rand.random() < WINDCHANCE

def update(pos, action):
	""" Updates position with given action. """
	# if wind():
	# 	pos += np.array(WINDSTRENGTH)
	return pos + (action * 0.01) 

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
	policy = Network(genome)
	if not z:
		z = np.random.uniform(0,1,2)
	pos = z
	l = [pos]
	ret = 0
	for i in range(max_steps+1):
		action = policy.propagate(list(pos),t=1)
		pos = update(pos, np.array(action))
		l.append(list(pos))
		if checkGoal(pos):
			ret = 0.9 ** i * 1000
			break
		if not checkBounds(pos):
			ret = 0.9 ** i * -1000
			break;
		ret = 0.9 ** i
	
	if verbose:
		draw(l)
	return ret
	
def score_function(x_predict,reward_predict,MSE, pi_amount, z_amount):
	"""
		Score function for selecting (pi,z)
		Returns a matrix of dimension len(reward_predict)*
		A score for pi and z for every row in x_predict and corresponding reward_predict
	"""

	# reshape results to grid
	reward_predictGrid = np.reshape(reward_predict, (pi_amount, z_amount))

	# get variance of Z over pi and reshape to score per pi-z pair
	var_z = np.var(reward_predictGrid, axis=0, keepdims=True)

	var_z -= var_z.min()
	if var_z.max() > 0:
		var_z /= var_z.max()

	# var_z *= weight_controllability_score
	
		
	# get mean of pi over Z and reshape to score per pi-z pair
	mean_pi = np.mean(reward_predictGrid, axis=1, keepdims=True)

	mean_pi -= mean_pi.min()
	if mean_pi.max() > 0:
		mean_pi /= mean_pi.max()

	MSE -= MSE.min()
	if MSE.max() > 0:
		MSE /= MSE.max()

	z_pi_score = 5 * np.ravel(mean_pi.dot(var_z))
	
	return 1.96 * MSE + z_pi_score

def do_evolution(pi_pool, z_pool , GP):
	"""
		Evolve the organisms and predict their values according to the GP
	"""
	# Evolve pools
	pi_pool = eonn.epoch(pi_pool,len(pi_pool))
	z_pool = eonn.epoch(z_pool,len(z_pool))
	
	# Create prediction matrix for GP
	x_predict = [np.append(pi_org.weights,z_org.weights) for pi_org in pi_pool for z_org in z_pool]
	
	# Get rewards and MSE
	reward_predict, MSE = GP.predict(x_predict, eval_MSE = True)
	
	return pi_pool, z_pool, x_predict, reward_predict, MSE

def add_scores(pi_pool, z_pool, x_predict, score_vector):
		"""
			Add the scores from the score_vector to the organisms in the pools
		"""
		# Append the scores to the evaluations of the organisms
		for i,score in enumerate(score_vector):
			
			# Get the organisms from the pools
			pi_org = pi_pool.find(x_predict[i][:-2])
			z_org = z_pool.find(x_predict[i][-2:])
			
			pi_org.evals.append(score)
			z_org.evals.append(score)

def acquisition(GP, epochs):
	"""
		Select the best (pi,z)-pair to evaluate using GP and GA
	"""
	
	# Create a pool for the policies
	pi_pool = Pool.spawn(Genome.open(NN_STRUCTURE_FILE),20,std = 8)
	
	# Create a pool of z's, starting around [0.5,0.5], should probably be better
	z_list = list(itertools.product(np.arange(0.1,1,1/5),np.arange(0.1,1,1/4)))
	genomes = BasicGenome.from_list(z_list, 20)
	org_list = [Organism(genome) for genome in genomes]
	z_pool  = Pool(org_list)
	
	for _ in xrange(epochs):
		pi_pool, z_pool, x_predict, reward_predict, MSE = do_evolution(pi_pool, z_pool, GP)
		# Get the scores according to the score function

		score_vector = score_function(x_predict, reward_predict, MSE, len(pi_pool), len(z_pool))
		add_scores(pi_pool, z_pool, x_predict, score_vector)
	
	# Get the current best combination (pi,z) and return the organisms for those
	sorted_reward = np.argsort(score_vector)
	best_combination = x_predict[sorted_reward[-1]]
	
	pi_org = pi_pool.find(best_combination[:-2])
	z_org = z_pool.find(best_combination[-2:])
	
	return pi_org, z_org
	
def initGP():
	"""Do 2 simulations with random pi,z and create GP, X, y"""
	pool = Pool.spawn(Genome.open(NN_STRUCTURE_FILE), 68, std = 10)
	X = []
	for org in pool:
		for gene in org.genome:
			gene.mutate()
		genome = org.genome
		w = genome.weights
		z = list(np.random.uniform(0,1,2))
		reward = cliff(genome,z)
	
		if not len(X):
			X = np.atleast_2d(w+z)
			y = np.atleast_2d([reward])
		else:
			X = np.append(X,[w+z],axis = 0)
			y = np.append(y, [reward])
	
	#Maybe use other kernel?
	
	GP = GaussianProcess(regr = 'quadratic', corr = 'squared_exponential')
	GP.fit(X,y)
	
	return GP,X,y

def updateGP(GP,X,y,w,z,reward):
	X = np.append(X,[w+z],axis = 0)
	y = np.append(y, [reward])
	GP.fit(X,y)
	return GP, X, y


def find_best(GP):
	print "Finding the best policy"
	
	epochs = 100
	
	pool = Pool.spawn(Genome.open(NN_STRUCTURE_FILE),50, std = 8)
	all_z = list(itertools.product(np.linspace(0.1, 0.9, 9), repeat=2))
	for i in xrange(epochs):
		pool = eonn.epoch(pool, len(pool))
		for org in pool:
			for z in all_z:
				reward = GP.predict(np.append(org.weights, z))
				org.evals.append(reward[0])
		if i % 10 == 0:
			print "Avg fitness:\t",i,"evaluations:\t", "%.1f" % pool.fitness
	champion = max(pool)
	
	return champion

def main():
	""" Main function. """
	np.set_printoptions(precision=3)
	GP,X,y = initGP()
	
	for i in xrange(1,200):
		pi_org, z_org = acquisition(GP, int(math.sqrt(i)) * 5)
		z = z_org.weights
		
		reward = cliff(pi_org.genome, z)
		print "Evaluation:\t", "%d\t" % i, "Return:\t", "%.1f\t" % reward, np.array(z)

		w = pi_org.genome.weights
		GP,X,y = updateGP(GP,X,y,w,z,reward)
		
	champion = find_best(GP)
	r = []
	for i in range(100):
		r.append(cliff(champion.genome, verbose = True))
	print sum(r)/len(r)
	plt.show()

if __name__ == '__main__':
	main()

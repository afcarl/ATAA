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
import json
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

WINDSTRENGTH = [0,0]
 
NN_STRUCTURE_FILE = 'cliff.net'

def update(pos, action):
	""" Updates position with given action. """
	return pos + (action * 0.01) 

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
	no_wind_yet = True
	policy = Network(genome)
	if not z:
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
			break
		if no_wind_yet and pos[0] > 0.5:
			pos = gust_of_wind(pos)
			no_wind_yet = False
		if not checkBounds(pos):
			ret = - 0.99 ** i * 100
			break;
	
	if verbose:
		draw(l)
	return ret
	

def score_pi(reward_predictGrid, ub_predictGrid):
	"""
	Returns a reward for each z. Assumes a certain ordering in 
	the scores of reward_predict
	"""

	reward_predictGrid += ub_predictGrid

	mean_pi = np.mean(reward_predictGrid, axis=1)

	if mean_pi.max() > 0:
		mean_pi /= mean_pi.max()

	return mean_pi


def score_z(reward_predictGrid, ub_predictGrid):
	"""
	Returns a reward for each z. Assumes a certain ordering in 
	the scores of reward_predict.
	"""

	reward_predictGrid += ub_predictGrid

	# get variance of Z over pi and reshape to score per pi-z pair
	var_z = np.var(reward_predictGrid + ub_predictGrid, axis=0)

	if var_z.max() > 0:
		var_z /= var_z.max()

	return var_z

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


def add_z_scores(z_pool, x_predict, z_score):
	"""
	Adds the z_score of each z to the z organism
	in the pool. Uses x_predict to find the correct organism
	"""

	# make sure for exactly each z a score is given
	assert(len(z_pool) == len(z_score));
	# make sure the ordering assumed is correct 
	# z of 0th element and len(z)th element should be same
	assert((x_predict[0][-1:] == x_predict[len(z_score)][-1:]).all())

	for i,score in enumerate(z_score):
		# Get the z organism from the pool
		z_org = z_pool.find(x_predict[i][-1:])
		z_org.evals.append(score)


def add_pi_scores(pi_pool, x_predict, pi_score):
	"""
	Adds the pi_score of each pi to the pi organism
	in the pool. Uses x_predict to find the correct organism
	"""

	# make sure for exactly each pi a score is given
	assert(len(pi_pool) == len(pi_score));
	# make sure the ordering assumed is correct 
	# pi of 0th element and 1st element should be same (up to len(z)-1 element)
	assert((x_predict[0][:-1] == x_predict[1][:-1]).all())

	z_amount = len(x_predict) // len(pi_pool)

	for i,score in enumerate(pi_score):
		# Get the organisms from the pools, where i*z_amount is the ith pi
		pi_org = pi_pool.find(x_predict[i * z_amount][:-1])
		
		pi_org.evals.append(score)

def acquisition(GP, epochs):
	"""
		Select the best (pi,z)-pair to evaluate using GP and GA
	"""
	
	# Create a pool for the policies
	pi_pool = Pool.spawn(Genome.open(NN_STRUCTURE_FILE),20,std = 8)
	

	# Create a pool of z's, starting around [0.5,0.5], should probably be better
	z_list = list(itertools.product(np.arange(0,0.5,1./20)))

	genomes = BasicGenome.from_list(z_list, 20)
	org_list = [Organism(genome) for genome in genomes]
	z_pool  = Pool(org_list)

	for _ in xrange(epochs):
		pi_pool, z_pool, x_predict, reward_predict, MSE = do_evolution(pi_pool, z_pool, GP)

		# get scores
		reward_predictGrid = np.reshape(reward_predict, (len(pi_pool), len(z_pool)))

		ub = 1.96 * np.sqrt(MSE)

		ub_predictGrid = np.reshape(ub, (len(pi_pool), len(z_pool)))

		pi_score = score_pi(reward_predictGrid, ub_predictGrid)
		z_score = score_z(reward_predictGrid, ub_predictGrid)

		# add scores to organisms

		add_pi_scores(pi_pool, x_predict, pi_score)
		add_z_scores(z_pool, x_predict, z_score)

	
	# return current best pi and z
	pi_org = max(pi_pool)
	z_org = max(z_pool)

	return pi_org, z_org
	
def initGP():
	"""Do 2 simulations with random pi,z and create GP, X, y"""
	poolsize = 68
	pool = Pool.spawn(Genome.open(NN_STRUCTURE_FILE), poolsize, std = 10)
	X = []
	for org in pool:
		org.mutate()
		genome = org.genome
		w = genome.weights
		z = [np.random.uniform(0,0.5)]
		reward = cliff(genome,z)

		while reward <= 0 and len(X) < poolsize/2:
			for gene in org.genome:
				gene.mutate()
			org.mutate()
			genome = org.genome
			w = genome.weights
			z = [np.random.uniform(0,0.5)]
			reward = cliff(genome,z)
	
		if not len(X):
			X = np.atleast_2d(w+z)
			y = np.atleast_2d([reward])
		else:
			X = np.append(X,[w+z],axis = 0)
			y = np.append(y, [reward])
	
	#Maybe use other kernel?
	GP = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)

	GP.fit(X,y)
	
	return GP,X,y

def updateGP(GP,X,y,w,z,reward):
	X = np.append(X,[w+z],axis = 0)
	y = np.append(y, [reward])
	GP.fit(X,y)
	return GP, X, y


def find_best(GP):
	
	epochs = 100
	
	pool = Pool.spawn(Genome.open(NN_STRUCTURE_FILE),50, std = 8)
	all_z = list(np.linspace(0, 0.5, 10))
	for n in xrange(epochs):
		if n != 0:
			pool = eonn.epoch(pool, len(pool))
		weights = [np.append(org.weights,z) for org in pool for z in all_z]
		reward = GP.predict(weights)
		for i in xrange(len(pool)):
			pool[i].evals = list(reward[i*len(all_z):(i+1)*len(all_z)])

	champion = max(pool)
	
	return champion

def find_best_lower(GP):
	
	epochs = 100
	
	pool = Pool.spawn(Genome.open(NN_STRUCTURE_FILE),50, std = 8)
	all_z = list(np.linspace(0, 0.5, 10))
	for n in xrange(epochs):
		if n != 0:
			pool = eonn.epoch(pool, len(pool))
		weights = [np.append(org.weights,z) for org in pool for z in all_z]
		reward, MSE = GP.predict(weights, eval_MSE = True)
		reward -= 1.96 *  np.sqrt(MSE)
		for i in xrange(len(pool)):
			pool[i].evals = list(reward[i*len(all_z):(i+1)*len(all_z)])

	champion = max(pool)
	
	return champion

def find_best_upper(GP):
	
	epochs = 100
	
	pool = Pool.spawn(Genome.open(NN_STRUCTURE_FILE),50, std = 8)
	all_z = list(np.linspace(0, 0.5, 10))
	for n in xrange(epochs):
		if n != 0:
			pool = eonn.epoch(pool, len(pool))
		weights = [np.append(org.weights,z) for org in pool for z in all_z]
		reward, MSE = GP.predict(weights, eval_MSE = True)
		reward += 1.96 * np.sqrt(MSE)
		for i in xrange(len(pool)):
			pool[i].evals = list(reward[i*len(all_z):(i+1)*len(all_z)])

	champion = max(pool)
	
	return champion

def main():
	""" Main function. """

	epochs = 20
	np.set_printoptions(precision=3)
	GP,X,y = initGP()
	r_avg = np.empty(10)
	pred_avg = np.empty(10)
	r_lower = np.empty(10)
	pred_lower = np.empty(10)
	r_upper = np.empty(10)
	pred_upper = np.empty(10)
	for i in xrange(10):
		for j in xrange(i*epochs, (i+1) * epochs):
			pi_org, z_org = acquisition(GP, 100)
			z = z_org.weights
			reward = cliff(pi_org.genome, z)
			print "Evaluation:\t", "%d\t" % (j+1), "Return:\t", "%.1f\t" % reward, np.array(z)
			w = pi_org.genome.weights
			GP,X,y = updateGP(GP,X,y,w,z,reward)

		champion_average = find_best(GP)
		champion_lower = find_best_lower(GP)
		champion_upper = find_best_upper(GP)
		
		r = np.empty(10)

		all_z = list(np.linspace(0, 0.5, 10))
		for j,z in enumerate(all_z):
			r[j] = cliff(champion_average.genome, z = [z])
		r_avg[i] = np.average(r)
		for j,z in enumerate(all_z):
			r[j] = cliff(champion_lower.genome, z = [z])
		r_lower[i] = np.average(r)
		for j,z in enumerate(all_z):
			r[j] = cliff(champion_upper.genome, z = [z])
		r_upper[i] = np.average(r)
		pred_avg[i] = champion_average.fitness
		pred_lower[i] = champion_lower.fitness
		pred_upper[i] = champion_upper.fitness
	
	plt.plot(r_avg)
	plt.plot(pred_avg)
	plt.plot(r_lower)
	plt.plot(pred_lower)
	plt.plot(r_upper)
	plt.plot(pred_upper)
	plt.show()
	plt.clf()

	data = {'r_avg' : list(r_avg), 'pred_avg' : list(pred_avg), 'r_lower' : list(r_lower), 'pred_lower' : list(pred_lower), 'r_upper' : list(r_upper), 'pred_upper' : list(pred_upper)}
	with open('data.json','w') as w:
		json.dump(data,w)

	champion = find_best(GP)
	r = []
	for _ in range(100):
		r.append(cliff(champion.genome, verbose = True))
	print sum(r)/len(r)
	plt.show()


if __name__ == '__main__':
	main()

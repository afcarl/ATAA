from eonn import eonn
from eonn.genome import Genome, BasicGenome
from eonn.organism import Pool
from eonn.network import Network
import numpy as np
from matplotlib import pyplot as plt
import random as rand
import itertools

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
GOALRADIUS = 0.05

WINDCHANCE = 0.01
WINDSTRENGTH = [0,-0.2]

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
	print "Return", ret
	if verbose or ret > 1:
		draw(l)
		plt.show()
	return ret
	
def score_function(x_predict,reward_predict,MSE, piAmount, zAmount):
	"""
		Score function for selecting (pi,z)
		Returns a matrix of dimension len(reward_predict)*
		A score for pi and z for every row in x_predict and corresponding reward_predict
	"""

	# TODO: proper tests whether reshaping and stuff is working correctly
	# currently reward_predict is producing similar numbers all over the place
	# which is weird!

	# reshape results to grid
	reward_predictGrid = np.reshape(reward_predict, (piAmount, zAmount))
	MSEGrid = np.reshape(MSE, (piAmount, zAmount))

	# get variance of Z over pi and reshape to score per pi-z pair
	varZ = np.zeros((zAmount,1))
	for i in xrange(zAmount):
		varZ[i] = np.var(reward_predictGrid[:][i])
	zScore = np.ravel(np.tile(varZ, piAmount))
	
	# get mean of pi over Z and reshape to score per pi-z pair
	meanPi = np.zeros((piAmount, 1))
	for i in xrange(piAmount):
		meanPi[i] = np.mean(reward_predictGrid[i][:])
	piScore = np.ravel(np.repeat(meanPi, zAmount))

	# normalize scores
	uncertaintyScore = MSE / np.max(np.abs(MSE))
	piScore = piScore / np.max(np.abs(piScore))
	zScore = zScore / np.max(np.abs(zScore))

	return uncertaintyScore + piScore + zScore


def doEvolution(pi_pool, z_pool , GP):
	"""
		Evolve the organisms and predict their values according to the GP
	"""
	# Evolve pools
	eonn.epoch(pi_pool,len(pi_pool))
	eonn.epoch(z_pool,len(z_pool))
	
	# Create prediction matrix for GP
	x_predict = [np.append(pi_org.weights,z_org.weights) for pi_org in pi_pool for z_org in z_pool]
	
	# Get rewards and MSE
	reward_predict, MSE = GP.predict(x_predict, eval_MSE = True)
	
	return pi_pool, z_pool, x_predict, reward_predict, MSE

def addScores(pi_pool,z_pool,score_matrix):
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

def acquisition(GP):
	"""
		Select the best (pi,z)-pair to evaluate using GP and GA
	"""
	epochs = 25
	
	# Create a pool for the policies
	pi_pool = Pool.spawn(Genome.open('cliff.net'),20,std = 8)
	
	# Create a pool of z's, startubg around [0.5,0.5], should probably be better
	z_pool  = Pool.spawn(BasicGenome.from_list([0.5,0.5]),20, std = 1, frac = 1)
	
	for _ in xrange(epochs):
		pi_pool, z_pool, x_predict, reward_predict, MSE = doEvolution(pi_pool, z_pool, GP)
		
		# Get the scores according to the score function
		score_matrix = score_function(x_predict,reward_predict,MSE, len(pi_pool), len(z_pool))
	
	# Get the current best combination (pi,z) and return the organisms for those
	sorted_reward = np.argsort(score_matrix)
	best_combination = x_predict[sorted_reward[-1]]
	
	pi_org = pi_pool.find(best_combination[:-2])
	z_org = z_pool.find(best_combination[-2:])
	
	return pi_org, z_org
	
def initGP():
	"""Do 2 simulations with random pi,z and create GP, X, y"""
	genome = Genome.open('cliff.net')
	genome.mutate( std = 8)
	w = genome.weights
	z = list(np.random.uniform(0,1,2))
	reward = cliff(genome,z)
	
	X = np.atleast_2d(w+z)
	y = np.atleast_2d([reward])
	
	genome.mutate()
	w = genome.weights
	z = list(np.random.uniform(0,1,2))
	reward = cliff(genome,z)
	
	X = np.append(X,[w+z],axis = 0)
	y = np.append(y, [reward])
	
	GP = GaussianProcess()
	GP.fit(X,y)
	
	return GP,X,y

def updateGP(GP,X,y,w,z,reward):
	X = np.append(X,[w+z],axis = 0)
	y = np.append(y, [reward])
	GP.fit(X,y)
	return GP, X, y

def main():
	""" Main function. """
	
	GP,X,y = initGP()
	
	for i in xrange(100):
		pi_org, z_org = acquisition(GP)
		print "Evaluation: ", i+1,
		z = z_org.weights
		reward = cliff(pi_org.genome, z)
		w = pi_org.genome.weights
		GP,X,y = updateGP(GP,X,y,w,z,reward)






if __name__ == '__main__':
	main()

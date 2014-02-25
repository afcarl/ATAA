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
An organism object holds a genome (genotype) and its corresponding network
(phenotype). It is also a wrapper class that provides extra fitness measures.

Classes:
organism	-- a wrapper class for a genome that provides fitness metrics.
pool			-- a collection of organisms.

"""

from genome import Genome
from network import Network
import numpy as np


def spawn(genome, size, frac=0.5, std=0.5):
	""" Generates a pool of organisms based on a prototype genome. """
	assert	size > 1
	pool = [Organism(genome) for i in range(size)]
	for org in pool[1:]:
		org.mutate(frac, std)
	return pool


class Organism(object):
	""" Wrapper class that provides fitness metrics. """
	def __init__(self, genome):
		self.genome = Genome(genome)
		self.policy = Network(self.genome)
		self.evals = list()

	def __cmp__(self, other):
		return cmp(self.fitness, other.fitness)

	def __str__(self):
		return '%.3f' % self.fitness

	def crossover(self, other):
		""" Return a new organism by recombining the parents. """
		return Organism(self.genome.crossover(other.genome))

	def mutate(self, frac=0.1, std=1.0, repl=0.25):
		""" Mutate the organism by mutating its genome. """
		self.genome.mutate(frac, std, repl)
		self.policy = Network(self.genome)

	def copy(self):
		""" Return a deep copy of this organism. """
		org = Organism(self.genome)
		org.evals = list(self.evals)
		return org

	def equals(self,values):
		for i,gene in enumerate(self.genome):
			if not gene.dna[-1] == values[i]:
				return False
		return True

	@property
	def fitness(self):
		""" Average return. """
		try:
			return sum(self.evals, 0.) / len(self.evals)
		except ZeroDivisionError:
			return 0.


class Pool(list):
	""" A collection of organisms. """
	def __init__(self, organisms):
		super(Pool, self).__init__([org.copy() for org in organisms])
		
	def genomes(self):
		""" Genomes of the population in matrix form"""
		X = np.empty([len(self),len(self[0].genome)])
		for n,org in enumerate(self):
			for i,gene in enumerate(org.genome):
				X[n][i] = gene.dna[-1]
		
	def find(self,values):
		for org in self:
			if org.equals(values):
				return org
		return None
			
			
	@property
	def fitness(self):
		""" Average fitness of the population. """
		return sum([org.fitness for org in self]) / len(self)

	@classmethod
	def spawn(cls, genome, size, frac=0.5, std=0.5):
		""" Generates a pool of organisms based on a prototype genome. """
		assert size > 1
		organisms = [Organism(genome) for i in range(size)]
		for org in organisms[1:]:
			org.mutate(frac, std)
		return Pool(organisms)

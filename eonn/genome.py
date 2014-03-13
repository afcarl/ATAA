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
A genome encodes the genetic information of an organism. It consist of a
sequence of genes, each of which encodes a single property of the organism in
its DNA. Besides providing a compact representation of an organism's genotype,
a genome features also two core genetic operators: crossover and mutation.

Classes:
gene	 -- a gene is a basic unit of heredity
genome -- all hereditary information of an organism encoded in multiple genes

"""

import random
import numpy as np

# Gene types
NEURON = 'n'
SYNAPSE = 's'

# Neuron DNA encoding
ID = 0
POS = 1
FUNC = 2
BIAS = 3

# Synapse DNA encoding
SRC = 0
DST = 1
WEIGHT = 2

# Neuron position
INPUT = 0
OUTPUT = 1
HIDDEN = 2

# Neural function types
SUM = 0
SIGMOID = 1
TANH = 2


class Gene(object):
	""" A basic genetic building block encoding a single property in DNA. """
	def __init__(self, type, dna):
		assert type in [NEURON, SYNAPSE]
		self.type = type
		self.dna = list(dna)

	def __cmp__(self, other):
		return cmp(self.key, other.key)

	def __eq__(self, other):
		return self.type == other.type and self.dna == other.dna

	def __str__(self):
		return (self.type + ' %d' * len(self.dna[:-1]) + ' %+.4f') % tuple(self.dna)
	
	def equals(self,value):
		return self.dna[-1] == value

	def mutate(self, std=1.0, repl=0.25):
		""" Mutating a gene involves altering its DNA (weight or bias). """
		delta = random.gauss(0, std)
		if random.random() > repl:
			delta += self.dna[-1]
		self.dna[-1] = delta

	def copy(self):
		""" Return a deep copy of this gene. """
		return Gene(self.type, list(self.dna))

	@property
	def key(self):
		return tuple([self.type] + self.dna[:-1])


class Genome(list):
	""" A genome is a collection of genes and encodes a neural network. """
	def __init__(self, genes):
		super(Genome, self).__init__([gene.copy() for gene in sorted(genes)])

	def __str__(self):
		return str.join('\n', map(lambda x: str(x), self))

	def mutate(self, frac=0.2, std=1.0, repl=0.25):
		""" Mutate this genome (effectively altering its behavior)

		Keyword arguments:
		frac	-- fraction of genes that can be mutated (probability)
		std		-- specifies the standard deviation of the mutation distribution
		repl	-- the probability that a gene is replaced when mutated

		"""
		for gene in self:
			if not (gene.type == NEURON and gene.dna[POS] == INPUT):
				if random.random() < frac:
					gene.mutate(std, repl)

	def crossover(self, other):
		""" Produce a new genome by recombining two parents.

		During crossover the parents' genes are aligned. Each gene of the new genome
		is then either the average of its parents or a copy of one of them.
		"""
		if [gene.key for gene in self] != [gene.key for gene in other]:
			raise ValueError, 'Topologies are incompatible for crossover'
		genes = list()
		for s, o in zip(self, other):
			if random.randint(0, 1):
				dna = map(lambda x, y: (x + y) / 2, s.dna, o.dna)
			else:
				dna = random.choice([s.dna, o.dna])
			genes.append(Gene(s.type, dna))
		return Genome(genes)

	def save(self, filename):
		""" Save this genome's configuration to the specified filename. """
		fh = open(filename, 'w')
		fh.write(str(self))
		fh.close()
		
	def copy(self):
		return Genome(self)

	@classmethod
	def open(cls, filename):
		""" Load a genome from the specified config file. """
		genes = list()
		for line in open(filename):
			if not line.startswith('#'):
				type, dna = line[0], [eval(v) for v in line[1:].split()]
				genes.append(Gene(type, dna))
		return Genome(genes)
	
	@property
	def weights(self):
		""" Returns the weights of an organism """
		w = [0]*len(self)
		for i,gene in enumerate(self):
			w[i] = gene.dna[-1]
		return w
		
class BasicGenome(list):
	""" A genome is a collection of genes and encodes a neural network. """
	def __init__(self, genes):
		super(BasicGenome, self).__init__([gene.copy() for gene in genes])

	def __str__(self):
		return " ".join([str(gene) for gene in self])

	def mutate(self, frac=0.2, std=1.0, repl=0.25):
		""" Mutate this genome (effectively altering its behavior)

		Keyword arguments:
		frac	-- fraction of genes that can be mutated (probability)
		std		-- specifies the standard deviation of the mutation distribution
		repl	-- the probability that a gene is replaced when mutated

		"""
		for gene in self:
			if random.random() < frac:
					gene.mutate(std)

	def crossover(self, other):
		""" Produce a new genome by recombining two parents.

		During crossover the parents' genes are aligned. Each gene of the new genome
		is then either the average of its parents or a copy of one of them.
		"""
		if not len(self) == len(other):
			raise ValueError, 'Topologies are incompatible for crossover'
		
		genes = list()
		for s, o in zip(self, other):
			if random.randint(0, 1):
				dna = (s.dna + o.dna) / 2.
			else:
				dna = random.choice([s.dna, o.dna])
			genes.append(BasicGene(dna))
		return BasicGenome(genes)

	def save(self, filename):
		""" Save this genome's configuration to the specified filename. """
		fh = open(filename, 'w')
		fh.write(str(self))
		fh.close()
		
	def copy(self):
		return BasicGenome(self)
	
	@classmethod
	def from_list(cls, genome_list, size = 1):
		""" Load a genome from a list of genomes """
		if size == 1:
			genes = list()
			for gene in genome_list:
				genes.append(BasicGene(gene))
			return BasicGenome(genes)
		else:
			l = []
			for genome in genome_list:
				genes = list()
				for gene in genome:
					genes.append(BasicGene(gene))
				l.append(BasicGenome(genes))
			return l
		
	@property
	def weights(self):
		""" Returns the weights of an organism """
		w = []
		for gene in self:
			w.append(gene.dna)
		return w
		
class BasicGene(object):
	""" A basic genetic building block encoding a single property in DNA. """
	def __init__(self, dna):
		self.dna = dna

	def __cmp__(self, other):
		return cmp(self.dna, other.dna)

	def __eq__(self, other):
		return self.dna == other.dna
		
	def equals(self,value):
		return self.dna == value

	def __str__(self):
		return str(self.dna)

	def mutate(self, std = 0.1):
		""" Mutating a gene involves altering its DNA (weight or bias). """
		delta = np.random.normal(0,1)
		
		if self.dna+delta <= 0 or self.dna+delta >= 1:
			self.mutate(std)
		else:
			self.dna += delta

	def copy(self):
		""" Return a deep copy of this gene. """
		return BasicGene(self.dna)
import numpy as np
import matplotlib.pyplot as plt

co_champ_fitness = np.load('champ_fitness_coevo.npy')
co_avg_fitness = np.load('avg_fitness_coevo.npy')
evo_champ_fitness = np.load('champ_fitness_evo.npy')
evo_avg_fitness = np.load('avg_fitness_evo.npy')

co_champ_fitness_mean = np.mean(co_champ_fitness, axis = 0)
co_champ_fitness_std = 1.96 * np.sqrt(np.var(co_champ_fitness, axis = 0))
evo_champ_fitness_mean = np.mean(evo_champ_fitness, axis = 0)
evo_champ_fitness_std = 1.96 * np.sqrt(np.var(evo_champ_fitness, axis = 0))

plt.plot(co_champ_fitness_mean, 'r', label = 'Co-Evolution')
plt.plot(co_champ_fitness_mean + co_champ_fitness_std, 'r--', label = 'Confidence Interval 95%')
plt.plot(co_champ_fitness_mean - co_champ_fitness_std, 'r--')

plt.legend(loc = 'lower right')
plt.ylim([-60,100])
plt.xlim([0,150])
plt.ylabel('Fitness')
plt.xlabel('Epochs')
plt.savefig('co_evo.png')
plt.clf()

plt.plot(evo_champ_fitness_mean, 'b', label = 'Evolution')
plt.plot(evo_champ_fitness_mean + evo_champ_fitness_std, 'b--', label = 'Confidence Interval 95%')
plt.plot(evo_champ_fitness_mean - evo_champ_fitness_std, 'b--')

plt.legend(loc = 'lower right')
plt.ylim([-60,100])
plt.xlim([0,150])
plt.ylabel('Fitness')
plt.xlabel('Epochs')
plt.savefig('evo.png')
plt.clf()

plt.plot(co_champ_fitness_mean, 'r', label = 'Co-Evolution')

plt.plot(evo_champ_fitness_mean, 'b', label = 'Evolution')

plt.legend(loc = 'lower right')
plt.ylim([-60,100])
plt.xlim([0,150])
plt.ylabel('Fitness')
plt.xlabel('Epochs')
plt.savefig('together.png')
plt.clf()
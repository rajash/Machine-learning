
import numpy as np 
from environment import Environment

class Route():

    def __init__(self, dnaLength):
        self.dnaLength = dnaLength
        self.dna = list()
        self.distance = 0

        for i in range(self.dnaLength - 1):
            rnd = np.random.randint(1, self.dnaLength)
            while rnd in self.dna:
                rnd = np.random.randint(1, self.dnaLength)
            self.dna.append(rnd)
        self.dna.append(0)

    def mix(self, dna1, dna2):
        self.dna = dna1.copy()

        for i in range(self.dnaLength - 1):
            if np.random.rand() <= 0.5:
                previous = self.dna[i]
                inx = self.dna.index(dna2[i])
                self.dna[inx] = previous
                self.dna[i] = dna2[i]

        for i in range(self.dnaLength - 1):
            if np.random.rand() <= 0.1:
                previous = self.dna[i]
                rnd = np.random.randint(1, self.dnaLength)
                inx = self.dna.index(rnd)
                self.dna[inx] = previous 
                self.dna[i] = rnd

            elif np.random.rand() <= 0.1:
                rnd = np.random.randint(1, self.dnaLength)
                preInx = self.dna.index(rnd)
                self.dna.insert(i, rnd)

                if i >= preInx:
                    self.dna.pop(preInx)
                else:
                    self.dna.pop(preInx + 1)

populationSize = 50
mutationRate = 0.1
nSelected = 5

env = Environment()
dnaLength = len(env.planets)
population = list()

for i in range(populationSize):
    route = Route(dnaLength)
    population.append(route)

generation = 0 
bestDist = np.inf
while True:
    generation += 1

    for route in population:
        env.reset()

        for i in range(dnaLength):
            action = route.dna[i]

            route.distance += env.step(action, 'none')
    
    sortedPop = sorted(population, key = lambda x: x.distance)
    population.clear()

    if sortedPop[0].distance < bestDist:
        bestDist = sortedPop[0].distance

    for i in range(nSelected):
        best = sortedPop[i]
        best.distance = 0
        population.append(best)

    left = populationSize - nSelected

    for i in range(left):
        newRoute = Route(dnaLength)
        if np.random.rand() <= mutationRate:
            population.append(newRoute)
        else:
            inx1 = np.random.randint(0, nSelected)
            inx2 = np.random.randint(0, nSelected)
            while inx1 == inx2:
                inx2 = np.random.randint(0, nSelected)
            
            dna1 = sortedPop[inx1].dna
            dna2 = sortedPop[inx2].dna

            newRoute.mix(dna1, dna2)

            population.append(newRoute)

    env.reset()

    for i in range(dnaLength):
        action = sortedPop[0].dna[i]
        _ = env.step(action, 'normal')

    if generation % 100 == 0:
        env.reset()

        for i in range(dnaLength):
            action = sortedPop[0].dna[i]
            _ = env.step(action, 'beautiful')

    print('Generation: '+ str(generation) + 'Shortest distance: {:.2f}'.format(bestDist) + 'light years')




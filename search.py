import torch

class EvolutionSearcher(object):

    def __init__(self, args):
        self.population_num = 50
        self.select_num = 10
        self.mutation_num = 25
        self.crossover_num = 25
        self.max_epochs = 20
    

    def search(self):
        print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        

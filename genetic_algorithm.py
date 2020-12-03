import numpy as np
np.random.seed(1)



class GeneticAlgorithm:
    
    def __init__(self, link_lengths, start_cood, end_cood, obs_coods, fitness, mu=[0.4,0.2], epsilon=0.1, population_size=120, mutation_percent=0.05, crossover_percent=0.30, generations=500):
        self.L1 = link_lengths[0]
        self.L2 = link_lengths[1]
        
        self.start_cood = start_cood
        self.end_cood = end_cood
        self.obs_coods = obs_coods
        self.fitness = fitness # fitenss function, takes population as input
        self.mu = mu
        self.epsilon = epsilon

        self.fitness_params = (link_lengths, start_cood, end_cood, obs_coods, epsilon, mu)

        self.L = 12
        self.x_min = 0                   
        self.x_max = pow(2,self.L)-1          
        self.y_min = 0                   
        self.y_max = pow(2,self.L)-1          
        
        self.R1 = sum(link_lengths)
        self.R2 = self.L1
        if self.R2 < 0: self.R2 = 0

        self.population_size = population_size
        self.mutation_percent = mutation_percent
        self.crossover_percent = crossover_percent
        self.generations = generations
        self.k = self.n_obstacles_interior() + 1

        self.fitness_stats = []


    def n_obstacles_interior(self):
        if len(self.obs_coods) == 0:
            return 0
        obs_coods = np.array(self.obs_coods)
        x_interior = obs_coods[:,0]
        y_interior = obs_coods[:,1]
        distance = np.sqrt(x_interior**2+y_interior**2) 
        
        distance = (distance<self.R1)
        return len(distance[distance==True])


    def chromosome_to_points(self, chromosome):
        return (chromosome)*(2*self.R1)/(2**self.L-1) - self.R1


    def chromosome_init(self):
        chromosome = np.zeros((self.population_size,2*self.k))

        centre_cood = [2**(self.L-1)-1, 2**(self.L-1)-1]
        distance_max = (self.x_max+1)/2
        distance_min = self.R2/self.R1*distance_max
        
        for i,chrom in enumerate(chromosome):
            chrom_valid = False
            
            while chrom_valid == False:
                random_chrom_x = np.random.randint(2**self.L,size=[self.k])
                random_chrom_y = np.random.randint(2**(self.L-1),size=[self.k]) + 2**(self.L-1)
                random_chrom = np.column_stack((random_chrom_x,random_chrom_y))

                distance = np.sqrt((random_chrom[:,0]-centre_cood[0])**2+(random_chrom[:,1]-centre_cood[1])**2)

                if np.all(distance_min < distance) and np.all(distance_max > distance):
                    chrom_valid = True
                    chromosome[i] = np.ravel(random_chrom)

        return chromosome


    def fitness_mod(self,chromosome):
        fitness_row, _ = self.fitness(self.chromosome_to_points(chromosome), *self.fitness_params)
        for i,v in enumerate(fitness_row):
            if np.isnan(v):
                fitness_row[i] = 0
            else:
                fitness_row[i] = abs(v)
        return fitness_row


    def run(self):
        chromosome = self.chromosome_init()
        fitness_row = self.fitness_mod(chromosome)
        r=1
        for genr in range(self.generations):
            
            print("New generation: "+str(r), end="\n", flush=True)
            r=r+1
            roulette_wheel_cdf = np.cumsum(fitness_row/np.sum(fitness_row))    
            crossover_point = np.random.randint(self.k-1) if self.k != 1 else 0                       
            index = np.zeros((2))
            new_chromosome = np.zeros((self.population_size,2*self.k))

            for i in range(int(self.population_size/2)):                 

                a = np.random.rand(2)
                index = np.searchsorted(roulette_wheel_cdf, a)  
                parent = np.array([chromosome[index[0]], chromosome[index[1]]])

                if np.random.rand() < self.crossover_percent:
                    new_chromosome[2*i+0,0:2*crossover_point+1] = parent[0,0:2*crossover_point+1]
                    new_chromosome[2*i+0,2*crossover_point+1:2*self.k] = parent[1,2*crossover_point+1:2*self.k]
                    new_chromosome[2*i+1,0:2*crossover_point+1] = parent[1,0:2*crossover_point+1]
                    new_chromosome[2*i+1,2*crossover_point+1:2*self.k] = parent[0,2*crossover_point+1:2*self.k]
                else:
                    new_chromosome[2*i+0] = parent[0]
                    new_chromosome[2*i+1] = parent[1]


            for i in range(self.population_size):                                 
                if (np.random.rand() < self.mutation_percent):
                    p = np.random.randint(12*2*self.k)
                    q = int(np.floor(p/12))
                    p = int(p - 12*q)

                    binary = list(np.binary_repr(int(new_chromosome[i,q]),12))
                    if (binary[p] == '0'):
                        binary[p] = '1'
                    elif (binary[p] == '1'):
                        binary[p] = '0'
                    binary_string = "".join(binary)            
                    new_chromosome[i,q] = int(binary_string,2)

            chromosome = new_chromosome
            fitness_row = self.fitness_mod(chromosome)
            self.fitness_stats.append(max(fitness_row))
            
        print()

        fitness_row = self.fitness_mod(chromosome)
        max_idx = np.argmax(fitness_row)
        return (self.chromosome_to_points(chromosome))[max_idx]
    

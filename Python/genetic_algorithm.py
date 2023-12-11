import numpy as np


def test_1(x):
    '''
    simple 1D test function
    '''
    
    return x[0] ** 2

def test_2(x):
    return x[0] ** 2 + x[1] ** 2

def uniform_random_sampler(N , bounds):
    '''
    Function to generate random samples.
    
    Parameters
    ----------
    N : int
        Number of samples to generatet
    Bounds : array
        Each element of the array should be a 
        tuple (or list/array) with length 2, that
        defines the limits of each parameter
    '''
    
    result = []
    
    for b in bounds:
    
        sample = np.random.uniform(b[0] , b[1] , N)
        result.append(sample)
        

    result = np.array(result)
    return np.transpose(result)


def eval_pop(f , population):
    '''
    simple function to evaluate the fitness of all of the
    members of the given population.
    Parameters
    ----------
    f : function
        evaluation function
    population : array
        Current population
    '''
    fitness = []
    for sltn in population:
        fitness.append(f(sltn))
    return np.array(fitness)

def reproduce(f , bounds , population, fitness , N_new):
    
    '''
    Function to take in the current population, and
    to create the next generation. Population size
    will be preserved. It is assumed that the
    population is sorted in order of decreasing fitness
    
    Parameters
    ----------
    f : function
        evaluation function
    bounds : array
        parameter boundaries
    population : array
        current population
    N_new : int
        number of new creatures to create
    '''
    
    new_sltns = []
    
    for i in range(N_new):
        nx = []
        parents = np.random.choice(np.array(range(N_new)) , 2 , replace = False )
        parent1 = population[parents[0]]
        parent2 = population[parents[1]]
       
        for k in range(len(parent1)):
            p1 = parent1[k]
            p2 = parent2[k]
            
            nparam = np.random.normal((p1 + p2) / 2.0 , abs(p2 - p1) , 1)[0]

            while nparam < bounds[k][0] or nparam > bounds[k][1]:
                ##Generate new sltns until we find one withing the bounds
                nparam = np.random.normal((p1 + p2) / 2.0 , abs(p2 - p1) , 1)[0]

            nx.append(nparam)
            
        new_sltns.append(nx)

    new_sltns = np.array(new_sltns)


    new_fitness = eval_pop(f , new_sltns)
    ii = np.arange(len(population) - N_new , len(population) , 1 )
    
    
    population[ii] = new_sltns
    fitness[ii] = new_fitness
    #fitness = eval_pop(f , population)
    
    ii = np.argsort(fitness)
    
    return population[ii] , fitness[ii]

def opt_w_random_samples(f , bounds , N):
    
    '''
    Tries to optimize a function by random sampling across
    the domnain. This is highly inefficient of course, but
    serves as a base comparison for better algorithms.
    '''
    
    result = uniform_random_sampler(N , bounds)
    f_vals = []
    x_vals = []
    for x in result:
        f_vals.append(f(x))
        x_vals.append(x)
    
    min_i = f_vals.index(min(f_vals))
    print ("Best f(x):" , f_vals[min_i])
    print ("Best x :" , result[min_i])
    return result[min_i] , f_vals[min_i]

def genetic_algorithm(f , bounds , popsize = 10 , Niter = 100, Gensize = 0.5):
    '''
    This is the main function that runs the actual
    genetic algorithm
    '''

    new = int(Gensize * popsize)
    ##First step is to initialize a population of solutions
    population = uniform_random_sampler(popsize , bounds)
    
    ##Determine fitness, and sort in order of increasing f eval
    fitness = eval_pop(f , population)
    ii = np.argsort(fitness)
    fitness = fitness[ii]
    population = population[ii]
    
    ##Iterate Niter times, generating new populations
    for i in range(Niter):
        population , fitness = reproduce(f,bounds,population,fitness,new)
    
    print ("Best f(x) :", fitness[0])
    print ("Best x:" , population[0])
    return population[0] , fitness[0]
    
    
if __name__ == "__main__":
    x1 , f1 = opt_w_random_samples(test_1 , [(-10,10)] , 200000)
    x2 , f2 = opt_w_random_samples(test_2 , [(-10,10),(-10,10)] , 200000)
    
    xg , fg = genetic_algorithm(test_2 , [(-10,10),(-10,10)] , popsize = 50 , Niter = 100)
    
    print ("Genetic Algoritm fitness / random --> ", fg / f2)
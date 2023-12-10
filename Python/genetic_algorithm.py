import numpy as np


def test_1(x):
    '''
    simple 1D test function
    '''
    
    return x[0] ** 2

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

if __name__ == "__main__":
    opt_w_random_samples(test_1 , [(-10,10)] , 1000)
    
    
'''
Code to replicate early N-body galaxy simulations.
Implements a basic tree-code based N-body simulation, wiht modular
support for stepping and timestep setting schemes.
'''

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

## Code Organization:

## There is a class to store our simulation data, called NBody
## This function contains the core functions for running a simulation,
## including the tree code. Keeps track of simulation units, and
## monitors the simulation state.

class NBody:
    
    def __init__(self , npart , dunit , vunit , munit , tunit , aunit , softl = 1.):
        self.N = npart
        self.x = []
        self.y = []
        self.z = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.mass = []
        self.dunit = dunit
        self.vunit = vunit
        self.munit = munit
        self.tunit = tunit
        self.aunit = aunit
        self.softl = softl
    

    def set_IC(self , x , y , z , vx , vy , vz , mass):
        '''
        Function to establish the initial conditions. Will set up
        simulation units, and prepare the full particle set.
        '''
        if len(x) != self.N:
            raise ValueError
        
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass
        
    def compute_accelerations(self , use_tree = True , use_soft = True):
        '''
        Function to calculate accelerations on all particles
        '''
        self.ax = []
        self.ay = []
        self.az = []
        aunit = (const.G * self.munit / (self.dunit **2 )).to(self.aunit).value
        print (aunit)
        
        
        if not use_tree:
            ## Exact solution, with bad scaling (N^2)

            for i in range(len(x)):
                ax = 0
                ay = 0
                az = 0
                
                for j in range(len(x)):
                    
                    if i == j:
                        continue
                    
                    rsq = ( ( self.x[i] - self.x[j] ) ** 2
                            + ( self.y[i] - self.y[j] ) ** 2
                            + (self.z[i] + self.z[j] ) ** 2 )
                    
                    amag = const.G.value * self.mass[j] / rsq
                    ax += amag / (self.x[j] - self.x[i])
                    ay += amag / (self.y[j] - self.y[i])
                    az += amag / (self.z[j] - self.z[i])
                
                self.ax.append(ax * aunit)
                self.ay.append(ay * aunit)
                self.az.append(az * aunit)

    def leap_frog(self , dt): 
        xi = self.x[-1] + self.vx[-1] * dt + 0.5 * self.ax[-1] * dt ** 2
        yi = self.y[-1] + self.vy[-1] * dt * 0.5 * self.ay[-1] * dt ** 2
        zi = self.z[-1] + self.vz[-1] * dt * 0.5 * self.az[-1] * dt ** 2
        
        
        
    def run_sim(self , dt , tstart , tend):
        t = tstart
        
        while t < tend:
            
            self.compute_accelerations(use_tree=False)
            self.leap_frog(dt)
if __name__ == "__main__":
    sim = NBody(100 , dunit = u.kpc , vunit = u.km / u.s , munit = u.Msun , tunit = u.Gyr, aunit = u.km / (u.s * u.Gyr))
    x = np.linspace(0 , 10 , 100)
    y = np.linspace(0 , 1 , 100)
    z = np.linspace(-0.1,0.1,100)
    vx = np.zeros(100)
    vy = np.zeros(100)
    vz = np.zeros(100)
    mass = np.ones(100)
    sim.set_IC(x,y,z,vx,vy,vz,mass)
    sim.compute_accelerations(use_tree = False)
    
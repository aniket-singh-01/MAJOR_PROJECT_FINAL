import numpy as np
import random
from tensorflow.keras.callbacks import Callback

class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer (GWO) for hyperparameter tuning
    """
    def __init__(self, 
                 param_space,
                 population_size=10, 
                 max_iter=30,
                 a_init=2,
                 a_final=0):
        """
        Initialize GWO
        
        Args:
            param_space: Dictionary of parameter ranges {param_name: (min, max)}
            population_size: Number of wolves in the pack
            max_iter: Maximum number of iterations
            a_init: Initial value of 'a'
            a_final: Final value of 'a'
        """
        self.param_space = param_space
        self.population_size = population_size
        self.max_iter = max_iter
        self.a_init = a_init
        self.a_final = a_final
        self.dim = len(param_space)
        self.param_names = list(param_space.keys())
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self):
        """Initialize wolf positions randomly within param_space"""
        population = []
        
        for _ in range(self.population_size):
            wolf = {}
            for param_name, (min_val, max_val) in self.param_space.items():
                # Handle different parameter types
                if isinstance(min_val, int) and isinstance(max_val, int):
                    wolf[param_name] = random.randint(min_val, max_val)
                elif isinstance(min_val, float) or isinstance(max_val, float):
                    wolf[param_name] = random.uniform(min_val, max_val)
                elif isinstance(min_val, list):
                    wolf[param_name] = random.choice(min_val)
            
            population.append(wolf)
        
        return population
    
    def optimize(self, fitness_func):
        """
        Run the GWO optimization
        
        Args:
            fitness_func: Function to evaluate fitness of solutions
            
        Returns:
            Best solution found and its fitness value
        """
        # Initialize population
        population = self.initialize_population()
        
        # Initialize alpha, beta, and delta wolves
        alpha = {param: 0 for param in self.param_names}
        beta = {param: 0 for param in self.param_names}
        delta = {param: 0 for param in self.param_names}
        
        alpha_score = float('inf')
        beta_score = float('inf')
        delta_score = float('inf')
        
        # Main loop
        for iter_idx in range(self.max_iter):
            # Update alpha, beta, and delta
            for i, wolf in enumerate(population):
                # Evaluate fitness
                fitness = fitness_func(wolf)
                
                # Update alpha, beta, delta
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha = wolf.copy()
                    
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = wolf.copy()
                        
                elif fitness < beta_score:
                    beta_score = fitness
                    beta = wolf.copy()
                    
                elif fitness < delta_score:
                    delta_score = fitness
                    delta = wolf.copy()
            
            # Linearly decrease parameter a from a_init to a_final
            a = self.a_init - iter_idx * ((self.a_init - self.a_final) / self.max_iter)
            
            # Update positions
            for i, wolf in enumerate(population):
                new_wolf = {}
                
                for param_name in self.param_names:
                    min_val, max_val = self.param_space[param_name]
                    
                    # Calculate position update
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    current_val = wolf[param_name]
                    
                    # Calculate D values (distance)
                    D_alpha = abs(C1 * alpha[param_name] - current_val)
                    D_beta = abs(C2 * beta[param_name] - current_val)
                    D_delta = abs(C3 * delta[param_name] - current_val)
                    
                    # Calculate new positions
                    X1 = alpha[param_name] - A1 * D_alpha
                    X2 = beta[param_name] - A2 * D_beta
                    X3 = delta[param_name] - A3 * D_delta
                    
                    # Average the three positions
                    new_val = (X1 + X2 + X3) / 3
                    
                    # Ensure new value is within bounds
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        new_val = int(round(new_val))
                        new_val = max(min_val, min(new_val, max_val))
                    elif isinstance(min_val, float) or isinstance(max_val, float):
                        new_val = float(new_val)
                        new_val = max(min_val, min(new_val, max_val))
                    elif isinstance(min_val, list):
                        # For categorical parameters, choose closest value
                        new_val = min(min_val, key=lambda x: abs(x - new_val))
                    
                    new_wolf[param_name] = new_val
                
                population[i] = new_wolf
            
            print(f"Iteration {iter_idx+1}/{self.max_iter}, Best fitness: {self.best_fitness}")
            
        return self.best_solution, self.best_fitness

def tune_hyperparameters(model_builder, train_data, val_data, param_space, n_epochs=5):
    """
    Use GWO to tune hyperparameters
    
    Args:
        model_builder: Function to build model with hyperparameters
        train_data: Training data generator
        val_data: Validation data generator
        param_space: Dictionary of parameter ranges
        n_epochs: Number of epochs for each evaluation
        
    Returns:
        Best hyperparameters found
    """
    def fitness_function(params):
        # Build model with the given hyperparameters
        model = model_builder(**params)
        
        # Train for a few epochs
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=n_epochs,
            verbose=0
        )
        
        # Return validation loss as fitness value (lower is better)
        return history.history['val_loss'][-1]
    
    # Initialize GWO and run optimization
    gwo = GreyWolfOptimizer(param_space=param_space)
    best_params, best_fitness = gwo.optimize(fitness_function)
    
    return best_params
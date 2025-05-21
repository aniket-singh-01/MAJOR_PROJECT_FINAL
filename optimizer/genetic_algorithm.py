import numpy as np
import random
from tensorflow.keras.models import Model
import tensorflow as tf

class GeneticAlgorithm:
    """
    Genetic Algorithm for feature selection
    """
    def __init__(self, 
                 population_size=20, 
                 n_generations=30,
                 mutation_rate=0.1,
                 crossover_rate=0.8):
        """
        Initialize GA
        
        Args:
            population_size: Size of the population
            n_generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_solution = None
        self.best_fitness = -np.inf
        
    def initialize_population(self, n_features):
        """Initialize random population of feature masks"""
        population = []
        
        for _ in range(self.population_size):
            # Create a binary mask for feature selection
            # Ensure at least one feature is selected
            chromosome = np.zeros(n_features, dtype=int)
            chromosome[np.random.choice(n_features, 
                                      size=np.random.randint(1, n_features+1),
                                      replace=False)] = 1
            population.append(chromosome)
        
        return population
    
    def selection(self, population, fitness_values):
        """Select parents using tournament selection"""
        selected = []
        
        for _ in range(len(population)):
            # Select 3 random individuals for tournament
            idx = np.random.choice(len(population), size=3, replace=False)
            tournament = [(i, fitness_values[i]) for i in idx]
            
            # Select the best one
            winner_idx = max(tournament, key=lambda x: x[1])[0]
            selected.append(population[winner_idx])
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Perform uniform crossover"""
        if random.random() < self.crossover_rate:
            n_features = len(parent1)
            child1, child2 = parent1.copy(), parent2.copy()
            
            # Uniform crossover
            for i in range(n_features):
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
            
            # Ensure at least one feature is selected
            if sum(child1) == 0:
                child1[random.randint(0, n_features-1)] = 1
            if sum(child2) == 0:
                child2[random.randint(0, n_features-1)] = 1
                
            return child1, child2
        else:
            return parent1, parent2
    
    def mutation(self, chromosome):
        """Perform bit-flip mutation"""
        mutated = chromosome.copy()
        n_features = len(chromosome)
        
        for i in range(n_features):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        
        # Ensure at least one feature is selected
        if sum(mutated) == 0:
            mutated[random.randint(0, n_features-1)] = 1
            
        return mutated
    
    def optimize(self, fitness_func, n_features):
        """
        Run the GA optimization
        
        Args:
            fitness_func: Function to evaluate fitness
            n_features: Total number of features
            
        Returns:
            Best solution and its fitness
        """
        # Initialize population
        population = self.initialize_population(n_features)
        
        for generation in range(self.n_generations):
            # Evaluate fitness of each individual
            fitness_values = [fitness_func(chromosome) for chromosome in population]
            
            # Update best solution
            max_fitness_idx = np.argmax(fitness_values)
            if fitness_values[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitness_values[max_fitness_idx]
                self.best_solution = population[max_fitness_idx].copy()
            
            # Selection
            selected = self.selection(population, fitness_values)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individual
            new_population.append(population[max_fitness_idx])
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
            
            print(f"Generation {generation+1}/{self.n_generations}, Best fitness: {self.best_fitness}")
            
        return self.best_solution, self.best_fitness

def feature_selection(model, X, y, layer_name):
    """
    Use GA to select important features from a specific layer
    
    Args:
        model: Trained model
        X: Input data
        y: True labels
        layer_name: Name of the layer to extract features from
        
    Returns:
        Feature mask (1 for selected, 0 for not selected)
    """
    # Create feature extractor model
    feature_extractor = Model(inputs=model.input, 
                            outputs=model.get_layer(layer_name).output)
    
    # Extract features
    features = feature_extractor.predict(X)
    n_features = features.shape[1]
    
    def fitness_function(feature_mask):
        # Select features based on mask
        selected_features = features[:, feature_mask == 1]
        
        # If no features selected, return lowest fitness
        if selected_features.shape[1] == 0:
            return -np.inf
        
        # Train a simple model on selected features
        inputs = tf.keras.Input(shape=(selected_features.shape[1],))
        x = Dense(64, activation='relu')(inputs)
        outputs = Dense(y.shape[1], activation='sigmoid')(x)
        temp_model = Model(inputs=inputs, outputs=outputs)
        
        temp_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for a few epochs
        history = temp_model.fit(
            selected_features, y,
            epochs=3,
            validation_split=0.2,
            verbose=0
        )
        
        # Return validation accuracy as fitness
        val_acc = history.history['val_accuracy'][-1]
        
        # Penalize for using too many features
        penalty = 0.001 * np.sum(feature_mask) / n_features
        
        return val_acc - penalty
    
    # Initialize GA and run optimization
    ga = GeneticAlgorithm()
    best_mask, best_fitness = ga.optimize(fitness_function, n_features)
    
    return best_mask
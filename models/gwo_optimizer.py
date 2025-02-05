import numpy as np

class GreyWolfOptimizer:
    def __init__(self, num_wolves, max_iter, search_space):
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        # Convert search space to numpy array
        self.search_space = np.array(search_space, dtype=float)
        
    def optimize(self, objective_function):
        # Initialize the grey wolf population
        wolves = np.random.uniform(
            low=self.search_space[:, 0],
            high=self.search_space[:, 1],
            size=(self.num_wolves, len(self.search_space))
        )
        
        # Initialize leaders with valid positions and fitness values
        alpha = [wolves[0], float('inf')]  # [position, fitness]
        beta = [wolves[1], float('inf')]   # [position, fitness]
        delta = [wolves[2], float('inf')]  # [position, fitness]
        
        a = 2  # Decreases linearly from 2 to 0
        
        for iter in range(self.max_iter):
            for i in range(self.num_wolves):
                # Update position
                for j in range(len(self.search_space)):
                    A1, C1 = self._compute_coefficients(a)
                    D_alpha = abs(C1 * alpha[0][j] - wolves[i][j])
                    X1 = alpha[0][j] - A1 * D_alpha
                    
                    A2, C2 = self._compute_coefficients(a)
                    D_beta = abs(C2 * beta[0][j] - wolves[i][j])
                    X2 = beta[0][j] - A2 * D_beta
                    
                    A3, C3 = self._compute_coefficients(a)
                    D_delta = abs(C3 * delta[0][j] - wolves[i][j])
                    X3 = delta[0][j] - A3 * D_delta
                    
                    wolves[i][j] = (X1 + X2 + X3) / 3
                
                # Evaluate fitness
                fitness = objective_function(wolves[i])
                
                # Update leaders
                if fitness < alpha[1]:
                    delta = beta
                    beta = alpha
                    alpha = [wolves[i], fitness]
                elif fitness < beta[1]:
                    delta = beta
                    beta = [wolves[i], fitness]
                elif fitness < delta[1]:
                    delta = [wolves[i], fitness]
            
            a = 2 - iter * (2 / self.max_iter)
        
        return alpha[0]

    def _compute_coefficients(self, a):
        A = 2 * a * np.random.random() - a
        C = 2 * np.random.random()
        return A, C 
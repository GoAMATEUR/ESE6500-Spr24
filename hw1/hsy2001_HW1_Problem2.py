import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations # (20,)
        self.Transition = Transition # (2, 2)
        self.Emission = Emission # (2, 3)
        self.Initial_distribution = Initial_distribution #(2,)

    def forward(self):
        alpha = np.zeros((self.Observations.shape[0], 
                          self.Transition.shape[0])) # (30, 2)
        alpha[0] = self.Initial_distribution * self.Emission[:, self.Observations[0]]
        for i in range(1, self.Observations.shape[0]):
            last_state = alpha[i-1]
            y_k = self.Observations[i]
            alpha[i] = self.Emission[:, y_k] * (last_state @ self.Transition)
        return alpha

    def backward(self):
        beta = np.zeros((self.Observations.shape[0],
                         self.Transition.shape[0]))
        beta[-1] = np.array([1., 1.])
        for i in range(1, self.Observations.shape[0]):
            last_state = beta[-i]
            y_k = self.Observations[-i]
            beta[-i-1] = self.Transition @ (last_state * self.Emission[:, y_k])
        return beta

    def gamma_comp(self, alpha, beta):
        gamma = np.multiply(alpha, beta) / np.sum(alpha, axis=-1)[:, np.newaxis] # alpha(num, x)
        gamma = gamma / np.sum(gamma, axis=-1)[:, np.newaxis]
        return gamma

    def xi_comp(self, alpha, beta, gamma):
        xi = np.multiply(alpha[..., np.newaxis], self.Transition) # t * x * x'
        M = self.Emission[:, self.Observations].T # t * x'
        xi = xi[:-1, :, :] * M[1:, np.newaxis, :] # t * x * x' * (t * 1 * x')
        xi = xi * beta[1:, np.newaxis, :] # (t-1) * x'
        xi = xi / np.sum(xi, axis=(1, 2))[:, np.newaxis, np.newaxis]
        return xi

    def update(self, alpha, beta, gamma, xi):
        new_init_state = gamma[0]
        # T_prime = np.zeros_like(self.Transition)
        # for i in range(2):
        #     g_sum = np.sum(gamma[:-1, i])
        #     for j in range(2):
        #         T_prime[i, j] = np.sum(xi[:, i, j]) / g_sum
        T_prime = np.sum(xi, axis=0) / np.sum(gamma[0:-1], axis=0)[:, np.newaxis]
        M_prime = np.zeros_like(self.Emission)
        for i in range(3):
            M_prime[:, i] = np.sum(gamma[np.where(self.Observations == i)], axis=0) / np.sum(gamma, axis=0)[np.newaxis, :]
        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):
        # print(alpha[-1])
        P_original = np.sum(alpha[-1])
        P_prime = 1
        new_p = HMM(self.Observations, T_prime, M_prime, new_init_state)
        new_alpha = new_p.forward()
        P_prime = np.sum(new_alpha[-1])
        return P_original, P_prime

if __name__ == "__main__":
    Mapping = {0: "LA", 1: "NY"}
    Emission = np.array([[0.4, 0.1, 0.5],
                         [0.1, 0.5, 0.4]])
    Transition = np.array([[0.5, 0.5],
                           [0.5, 0.5]])
    Observations = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])
    # print(Observations.shape)
    Initial_distribution = np.array([0.5, 0.5])
    
    p2 = HMM(Observations, Transition, Emission, Initial_distribution)
    alpha = p2.forward()
    beta = p2.backward()
    print(alpha)
    print(beta)
    gamma = p2.gamma_comp(alpha, beta)
    print(gamma)
    # print(np.argmax(gamma, axis=-1))
    xi = p2.xi_comp(alpha, beta, gamma) 
    # print(xi)
    T_prime, M_prime, new_init_state = p2.update(alpha, beta, gamma, xi)
    print(new_init_state)
    print(T_prime)
    print(M_prime)
    P_original, P_prime = p2.trajectory_probability(alpha, beta, T_prime, M_prime, new_init_state)
    print(P_original, P_prime)
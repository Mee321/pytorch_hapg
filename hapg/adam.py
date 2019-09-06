import numpy as np
import torch
class Adam_optimizer():
    def __init__(self, alpha, beta1, beta2, epsilon):
        self.lr = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.f_momentum = 0
        self.s_momentum = 0
    def update(self, gradient):
        self.f_momentum = self.beta1*self.f_momentum + (1-self.beta1)*gradient
        self.s_momentum = self.beta2*self.s_momentum + (1-self.beta2)*gradient*gradient
        fm = self.f_momentum / (1-self.beta1)
        sm = self.s_momentum / (1-self.beta2)
        direction = self.lr * fm / (torch.sqrt(sm) + self.epsilon)
        return direction
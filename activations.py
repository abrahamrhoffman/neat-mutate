#######################
# Mutate: Activations #
#######################
# Author: Abe Hoffman #
# Date  : Apr 2017    #
#######################

### Import Libraries ###
from __future__ import division
import math

### Activation Class ###
class Activation(object):

    def sigmoid_activation(self, z):
        z = max(-60.0, min(60.0, 5.0 * z))
        return 1.0 / (1.0 + math.exp(-z))

    def tanh_activation(self, z):
        z = max(-60.0, min(60.0, 2.5 * z))
        return math.tanh(z)

    def sin_activation(self, z):
        z = max(-60.0, min(60.0, 5.0 * z))
        return math.sin(z)

    def gauss_activation(self, z):
        z = max(-3.4, min(3.4, z))
        return math.exp(-5.0 * z**2)

    def relu_activation(self, z):
        return z if z > 0.0 else 0.0

    def softplus_activation(self, z):
        z = max(-60.0, min(60.0, 5.0 * z))
        return 0.2 * math.log(1 + math.exp(z))

    def identity_activation(self, z):
        return z

    def clamped_activation(self, z):
        return max(-1.0, min(1.0, z))

    def inv_activation(self, z):
        if z == 0:
            return 0.0
        else:
            return 1.0 / z

    def log_activation(self, z):
        z = max(1e-7, z)
        return math.log(z)

    def exp_activation(self, z):
        z = max(-60.0, min(60.0, z))
        return math.exp(z)

    def abs_activation(self, z):
        return abs(z)

    def hat_activation(self, z):
        return max(0.0, 1 - abs(z))

    def square_activation(self, z):
        return z ** 2

    def cube_activation(self, z):
        return z ** 3

    def abs_activation(z):
        return abs(z)

    def hat_activation(z):
        return max(0.0, 1 - abs(z))

    def square_activation(z):
        return z ** 2

    def cube_activation(z):
        return z ** 3

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
class Activations(object):

    def sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def tanh(self, z):
        return math.tanh(z)

    def sin(self, z):
        return math.sin(z)

    def gauss(self, z):
        return math.exp(-5.0 * z**2)

    def relu(self, z):
        return z if z > 0.0 else 0.0

    def softplus(self, z):
        return 0.2 * math.log(1 + math.exp(z))

    def identity(self, z):
        return z

    def clamped(self, z):
        return max(-1.0, min(1.0, z))

    def inv(self, z):
        if z == 0:
            return 0.0
        else:
            return 1.0 / z

    def log(self, z):
        return math.log(z)

    def exp(self, z):
        return math.exp(z)

    def abs(self, z):
        return abs(z)

    def hat(self, z):
        return max(0.0, 1 - abs(z))

    def square(self, z):
        return z ** 2

    def cube(self, z):
        return z ** 3

    def abs(self, z):
        return abs(z)

    def hat(self, z):
        return max(0.0, 1 - abs(z))

    def square(self, z):
        return z ** 2

    def cube(self, z):
        return z ** 3
from __future__ import print_function
import numpy as np
import torch

class Fitness(object):
    '''
    Fitness Function to evaluate the Phenome
    '''

    def __init__(self, data, PHENOME):
        X,Y = data
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.phenome = PHENOME

        if torch.cuda.is_available():
            self.GPUs = [GPU for GPU in range(torch.cuda.device_count())]

    def eval(self):
        try:
            result = torch.sigmoid(torch.matmul(self.X, self.phenome)).cuda(random.choice(self.GPUs))
            forwardError = np.sqrt(torch.mean(torch.pow((Y_ - result), 2).cuda()))
            return forwardError
        except:
            result = torch.sigmoid(torch.matmul(self.X, self.phenome))
            forwardError = np.sqrt(torch.mean(torch.pow((Y_ - result), 2)))
            return forwardError

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
        del PHENOME[-1]

        if torch.cuda.is_available():
            self.GPUs = [GPU for GPU in range(torch.cuda.device_count())]

    def layerDot(self, inputTensor, layerWeights):
        # Expect Torch Weights and Numpy Data (X,Y)
        try:
            inputTensor = inputTensor.numpy().copy()
        except:
            pass
        
        layerWeights = layerWeights.numpy().copy()
    
        # If the Input Tensor Column's value is the same as Layer1's Row value (compatibility):
        if inputTensor.shape[-1] == layerWeights.shape[0]:
            inputTensor = torch.from_numpy(inputTensor)
            layerWeights = torch.from_numpy(layerWeights)
            try:
                layerDotResult = torch.sigmoid(torch.matmul(inputTensor, layerWeights)).cuda(random.choice(self.GPUs))
            except:
                layerDotResult = torch.sigmoid(torch.matmul(inputTensor, layerWeights))
        else:
            # Reshape the layerWeights to be of compatible shape. Zero fill non-existent nodes/edges.
            shape = (inputTensor.shape[-1],layerWeights.shape[0])
            zeros = np.zeros(shape, dtype=np.int32)
            layerWeights.resize(shape,refcheck=False)
        
            inputTensor = torch.from_numpy(inputTensor)
            layerWeights = torch.from_numpy(layerWeights)
            
            try:
                layerDotResult = torch.sigmoid(torch.matmul(inputTensor, layerWeights)).cuda(random.choice(self.GPUs))
            except:
                layerDotResult = torch.sigmoid(torch.matmul(inputTensor, layerWeights))
    
        return layerDotResult            
            
            
    def eval(self):
        results = []
        for ix,layer in enumerate(self.phenome):
            if ix == 0:
                results.append(self.layerDot(self.X,layer))
            results.append(self.layerDot(results[ix-1],layer))
        
        try:
            forwardError = np.sqrt(torch.mean(torch.pow((self.Y - results[-1]), 2))).cuda(random.choice(self.GPUs))
        except:
            forwardError = np.sqrt(torch.mean(torch.pow((self.Y - results[-1]), 2)))
        
        return forwardError

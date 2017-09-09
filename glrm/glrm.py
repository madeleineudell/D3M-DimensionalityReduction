import julia
import numpy as np

"""
This class implements glrm, with default setting being PCA.
"""


#preprocess a string to optionally add a pair of round brackets afterwards
def string_preprocess(string):
    if string[-1] == ')': #the case in which input losses and regularizers already have parameters
        return string
    else:
        return string + '()'



class glrm:

    is_dimensionality_reduction = True
    hyperparameters = {}
    
    #class constructor: default being PCA
    def __init__(self, losses = 'QuadLoss', rx = 'ZeroReg', ry = 'ZeroReg'):
        
        self.losses = losses
        self.rx = rx
        self.ry = ry
        
        j = julia.Julia()
        j.using("LowRankModels")
    
    #make the losses and regularizers be either strings or lists
        if type(self.losses) == list:
            self.losses_j = [j.eval(string_preprocess(self.losses[i])) for i in range(len(self.losses))]
        else:
            self.losses_j = j.eval(string_preprocess(self.losses))

        if type(self.rx) == list:
            self.rx_j = [j.eval(string_preprocess(self.rx[i])) for i in range(len(self.rx))]
        else:
            self.rx_j = j.eval(string_preprocess(self.rx))

        if type(self.ry) == list:
            self.ry_j = [j.eval(string_preprocess(self.ry[i])) for i in range(len(self.ry))]
        else:
            self.ry_j = j.eval(string_preprocess(self.ry))

    def dimension_reduce(self, A, k):
        j = julia.Julia()
        j.using("LowRankModels")
        
        glrm_j = j.GLRM(A, self.losses_j, self.rx_j, self.ry_j, k)
        X, Y, ch = j.fit_b(glrm_j)
        self.Y = Y
        self.k = k
        
        return np.dot(np.transpose(X), Y)
        
    
    def output_map(self, v): #calculate the output map; requires the input to be an numpy array
        
        v = v.reshape(1, -1)
        
        j = julia.Julia()
        j.using("LowRankModels")
        
        #make sure dimension_reduce has already been executed beforehand, and Y and v match in terms of numbers of columns
        try:
            self.Y
        except NameError:
            raise Exception('Initial GLRM fitting not executed!')
        else:
            try:
                if self.Y.shape[1] != v.shape[1]:
                    raise ValueError
            except ValueError:
                raise Exception('Dimension of input vector does not match Y!')
            else:
                self.Y = self.Y.astype(float) #make sure column vectors finally have the datatype Array{float64,1} in Julia
                num_cols = self.Y.shape[1]
                ry_j = [j.FixedLatentFeaturesConstraint(self.Y[:, i]) for i in range(num_cols)]
                glrm_new_j = j.GLRM(v, self.losses_j, self.rx_j, ry_j, self.k)
                x, yp, ch = j.fit_b(glrm_new_j)
                return x


"""usage example"""
if __name__ == "__main__":
    
    # form an n x d array
    n = 10 # number of examples
    d = 5  # number of dimensions
    A = np.arange(n*d).reshape(n, d)
    k = 2
    
    g = glrm() #PCA
    #g = glrm(, 'NonNegConstraint', 'NonNegConstraint') #nnmf
    #g = glrm('HuberLoss', 'QuadReg') #rpca
    
    W = g.dimension_reduce(A, k)
    print(W)
    
    #calculate output map with respect to a new vector
    v = np.arange(d)
    r = g.output_map(v)
    print(r)
    

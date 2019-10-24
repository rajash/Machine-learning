class featureScale:
    def __init__(self):
        self.X = None
        self.mu = None
        self.sigma = None
        self.min = None
        self.max = None
    
    def set_param(self, X):
        X = np.array(X)
        self.X = X
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)

    def minMaxNormalize(self, X, same = False):
        if same == False:
               self.set_param(X)
        return (self.X - self.min) / (self.max - self.min)
    
    def meanNormalize(self, X, same = False):
        if same == False:
               self.set_param(X)
        return (self.X - self.mu) / (self.max - self.min)
    
    def standardize(self, X, same = False):
        if same == False:
               self.set_param(X)
        return (X - self.mu) / self.sigma
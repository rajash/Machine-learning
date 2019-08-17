class Regression:
    def __init__(self):
        self.theta = None
        self.X = None
        self.y = None
        self.num_examples = None
        self.history = None
        self.alpha = 0.01
        self.num_iters = 1000
        self.feature_scaler = None

    def perpareData(self, X):
        self.feature_scaler = featureScale()
        X = self.feature_scaler.standardize(X)
        self.X = np.append(arr = np.ones((X.shape[0],1)), values = X, axis = 1)

   
    def Cost(self, X, y, theta = None):
        if type(self.X) == type(None):
            self.perpareData(X)
        self.y = np.reshape(y,(y.shape[0] , -1))
        if type(theta) == type(None):
            if type(self.theta) == type(None):
                self.theta = np.zeros((self.X.shape[1],1))
        else:
            self.theta = theta
        self.num_examples = len(y)
        h = np.dot(self.X, self.theta)
        J = 0.5/self.num_examples * np.sum(np.dot((h-self.y).T, (h - self.y)), axis=0)
        return J
    
    def gradientDescent(self, X, y, theta = None, alpha = 0.01, num_iters = 1000):
        if type(self.X) == type(None):
            self.perpareData(X)
        self.y = np.reshape(y,(y.shape[0] , -1))
        if type(theta) == type(None):
            if type(self.theta) == type(None):
                self.theta = np.zeros((self.X.shape[1],1))
        else:
            self.theta = theta
        self.history = np.zeros((num_iters, 1))
        self.num_iters = num_iters
        self.num_examples = len(y)
        for i in range(self.num_iters):
            h = np.dot(self.X, self.theta)
            self.theta -= alpha/self.num_examples * np.reshape(np.sum(np.dot((h - self.y).T, self.X),axis = 0), (-1,1))
            self.history[i] = self.Cost(self.X, self.y, self.theta)
        return self.history
    
    def normalEquation(self, X, y):
        self.perpareData(X)
        self.y = np.reshape(y,(y.shape[0] , -1))
        self.num_examples = len(y)
        try:
            inversed = np.linalg.inv(np.dot(self.X.T, self.X))
        except np.linalg.LinAlgError:
            print('Warning: this array cannot be inverted.')
            return None
        else:
            self.theta = np.dot(inversed, np.dot(self.X.T,self.y))
            return self.theta
    
    def predict(self, newX):
        newX = np.array(newX)
        if(newX.ndim == 1):
            newX = np.reshape(newX,(1,-1))
        newX = np.append(arr = np.ones((newX.shape[0],1)), values = self.feature_scaler.standardize(newX, True), axis = 1)
        return np.dot(newX, self.theta)

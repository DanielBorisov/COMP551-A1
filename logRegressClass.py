import numpy as np

class LogisticRegression:
    # Class for performing logistic regression
    
    def __init__(self, nfeatures, decayPerEpoch=1, lam=1):
        self.decayPerEpoch = decayPerEpoch
        self.weights = np.zeros((nfeatures + 1, 1))
        self.nfeatures = nfeatures
        self.lam = lam

    def fit(self, x, y, rate, iterations, lrdecay, reg):
        # x and y is the training data
        # rate is the learning rate
        # iterations are the gradient descent iterations.|
        # featureMatrix = np.matlib.repmat(self.weights, 2, np.size(self.weights))
        x = np.c_[x, np.ones(np.size(x, 0))]
        for i in range(0, iterations):
            if lrdecay:
                rate -= 0.5 * rate * int((i % self.decayPerEpoch) == 0)

            a = np.dot(x, self.weights)
            sigma = 1/(1 + np.exp(-a))

            if reg:
                penalty = self.lam*np.sign(self.weights)
                wNext = self.weights.T + ((rate/self.nfeatures) * np.sum(x * (y - sigma), axis=0)) + penalty.T
                self.weights = wNext.T
            else:
                self.weights += (rate/self.nfeatures) * np.dot(x.T, (y - sigma))

        return self.weights

    def predict(self, x):
        #Predicts y
        #turns probabilities to binary, threshold at 0.5
        x = np.c_[x, np.ones(np.size(x, 0))]
        prediction = np.dot(x, self.weights)
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        return prediction
import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape         # This will return some output like (1000,10) rows columns
        self._classes = np.unique(y)            # This will return all the unique values in the Variables y
        n_classes = len(self._classes)          # Total no. of classes

        # Formula for Naive Bayes   p(Y/X) = {  p(x1/Y)*p(x2/Y)*p(x3/y) * p(y)    /      p(X)   }
        # Calculate mean, var, and prior for each class this will be used in the Probability density function
        # Initialisation of all the variables with zeros using the functions
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        # The Effect of the value of the predictor on the Given class

        for idx, c in enumerate(self._classes):
            X_columns = X[y == c]                     # Only 1000 Rows Taken and Per class
            self._mean[idx, :] = X_columns.mean(axis=0)
            self._var[idx, :] = X_columns.var(axis=0)
            self._priors[idx] = X_columns.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):                   #Probablity Density Functions
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

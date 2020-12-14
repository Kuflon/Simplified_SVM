import logging

import numpy as np

from baseestimator import BaseEstimator
from kernels import Linear

np.random.seed(9999)

class SVM(BaseEstimator):
    def __init__(self, C=1.0, kernel=None, tol=1e-3, max_iter=100):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

        if kernel is None:
            self.kernel = Linear()
        else:
            self.kernel = kernel

        self.b = 0
        self.K = None
        self.a = None
        self.memory = []
        self.multiclass = False

    def fit(self, X, y=None):
        self.X, self.y = self.set_input(X, y)

        classes = np.unique(self.y)
        self.classes = classes

        if classes.size == 2:
            return self._fit_two_classes(X, y)
        else:
            self.multiclass = True

            X, y = self.X, self.y
            for i in range(len(classes)-1):
                for j in range(i+1, len(classes)):
                    classifier = dict()
                    select_indices = np.where(np.logical_or(y == classes[i], y == classes[j]))
                    self.X = X[select_indices]
                    self.y = y[select_indices]
                    self.y[np.where(self.y == i)] = -1.0
                    self.y[np.where(self.y == j)] = 1.0

                    self._fit_two_classes(self.X, self.y)

                    classifier['sv_idx'] = self.sv_idx
                    classifier['a'] = self.a
                    classifier['b'] = self.b
                    classifier['y'] = self.y
                    classifier['X'] = self.X
                    classifier['class_1'] = i
                    classifier['class_2'] = j

                    self.b = 0
                    self.K = None
                    self.a = None

                    self.memory.append(classifier)


    def _fit_two_classes(self, X, y=None):
        self.set_input(X, y)
        self.K = np.zeros((self.n_samples, self.n_samples))

        for i in range(self.n_samples):
            self.K[:, i] = self.kernel(self.X, self.X[i, :])
            self.a = np.zeros(self.n_samples)
            self.sv_idx = np.arange(0, self.n_samples)

        return self._train()


    def _train(self):
        iter = 0

        while iter < self.max_iter:
            iter += 1
            a_old = np.copy(self.a)

            for j in range(self.n_samples):

                i = self.random_index(j)

                eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                if eta >= 0:
                    continue
                L, H = self._find_bounds(i, j)

                e_i, e_j = self._error(i), self._error(j)

                a_i_old, a_j_old = self.a[i], self.a[j]

                self.a[j] -= (self.y[j] * (e_i - e_j)) / eta
                self.a[j] = self._clip(self.a[j], H, L)

                self.a[i] += self.y[i] * self.y[j] * (a_j_old - self.a[j])

                b1 = (
                    self.b - e_i - self.y[i] * (self.a[i] - a_i_old) * self.K[i, i]
                    - self.y[j] * (self.a[j] - a_j_old) * self.K[i, j]
                )
                b2 = (
                    self.b - e_j - self.y[j] * (self.a[j] - a_j_old) * self.K[j, j]
                    - self.y[i] * (self.a[i] - a_i_old) * self.K[i, j]
                )
                if 0 < self.a[i] < self.C:
                    self.b = b1
                elif 0 < self.a[j] < self.C:
                    self.b = b2
                else:
                    self.b = 0.5 * (b1 + b2)

            diff = np.linalg.norm(self.a - a_old)
            if diff < self.tol:
                break
        logging.info("Convergence has reached after %s." % iter)

        # Save support vectors index
        self.sv_idx = np.where(self.a > 0)[0]



    def _predict(self, X=None):
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):

            if self.multiclass == True:
                result[i] = self._predict_multi(X[i, :])
            else:
                result[i] = np.sign(self._predict_row(X[i, :]))

        return result

    def _predict_multi(self, X):
        result_class = np.zeros(len(self.memory))
        result_value = np.zeros(len(self.memory))

        for i in range(len(self.memory)):
            self.sv_idx = self.memory[i]['sv_idx']
            self.a = self.memory[i]['a']
            self.b = self.memory[i]['b']
            self.y = self.memory[i]['y']
            self.X = self.memory[i]['X']
            result_class[i] = np.sign(self._predict_row(X))
            result_value[i] = self._predict_row(X)

        votes = self._count_votes(result_class, result_value)
        votes = list(votes.values())
        decide_class = np.argmax(votes)

        return decide_class

    def _count_votes(self, result_class, result_value):
        vote_per_class = dict()
        for name in self.classes:
            vote_per_class[name] = 0

        for i in range(len(self.memory)):
            if result_class[i] == -1:
                class_name = self.memory[i]['class_1']
            else:
                class_name = self.memory[i]['class_2']
            vote_per_class[class_name] +=1

        return vote_per_class

    def _predict_row(self, X):
        k_v = self.kernel(self.X[self.sv_idx], X)
        return np.dot((self.a[self.sv_idx] * self.y[self.sv_idx]).T, k_v.T) + self.b

    def random_index(self, k):
        i = k
        while i == k:
            i = np.random.randint(0, self.n_samples - 1)
        return i

    def _find_bounds(self, i, j):
        if self.y[i] != self.y[j]:
            L = max(0, self.a[j] - self.a[i])
            H = min(self.C, self.C + self.a[j] - self.a[i])
        else:
            L = max(0, self.a[i] + self.a[j] - self.C)
            H = min(self.C, self.a[i] + self.a[j])
        return L, H

    def _error(self, i):
        return self._predict_row(self.X[i]) - self.y[i]

    def _clip(self, alpha, H, L):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

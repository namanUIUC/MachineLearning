import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''


        binary_svm = {}
        #unique_label = np.unique(y)
        #lenth = unique_label.shape[0]
        for i in range(self.labels.shape[0]):

                y_temp = np.copy(y)
                y_temp[y[:] != self.labels[i]] = 0
                y_temp[y[:] == self.labels[i]] = 1
                clf = svm.LinearSVC(random_state=12345)
                clf.fit(X,y_temp)
                binary_svm[(self.labels[i])] = clf
        return binary_svm


    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        unique_label = np.unique(y)
        lenth = unique_label.shape[0]
        for i in range(lenth):
            for j in range(i+1,lenth):
                l1 = unique_label[i]
                l2 = unique_label[j]
                y_train = y[(y[:]==l1) | (y[:]==l2)]
                x_train = X[((y[:]==l1) | (y[:]==l2)),:]
                clf = svm.LinearSVC(random_state=12345)
                clf.fit(x_train,y_train)

                binary_svm[tuple((l1,l2))] = clf
        return binary_svm

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = []
        for key in self.labels:
            scores.append(self.binary_svm[key].decision_function(X))

        scores = np.array(scores)
        # print(np.transpose(scores).shape)
        return np.transpose(scores)



    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = []
        temp =[]
        for key, clf in self.binary_svm.items():
            temp.append(clf.predict(X))

        temp = np.transpose(np.array(temp))
        for i in range(temp.shape[0]):
            row = temp[i,:]
            scores.append(np.bincount(row,minlength=self.labels.shape[0]))
        return np.array(scores)


    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        l = np.unique(y)
        reg = np.sum(np.square(np.linalg.norm(W,axis=1)))
        slack = 0
        for i in range(X.shape[0]):
            temp = []
            for j in range(W.shape[0]):
                if (l[j]==y[i]):
                    loc = j
                    temp.append(np.dot(W[j,:], X[i,:]))
                else:
                    temp.append(1+np.dot(W[j,:], X[i,:]))
            temp = np.array(temp)
            max_val = np.max(temp)
            slack += max_val - np.dot(W[loc,:], X[i,:])

        return (0.5*reg + C*slack)

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        N = X.shape[0]
        K = W.shape[0]
        Delta = np.zeros((K,N))
        for j in range(K):
            for i in range(N):
                if j == y[i]:
                    Delta[j,i] = 1

        I = np.ones((K,N))
        sub = I - Delta + W.dot(X.T)

        # rsub = np.reshape(np.amax(sub, axis=0), (N,1))

        idx = np.argmax(sub, axis=0)
        max_grad = np.zeros_like(W)
        for num, val in enumerate(idx):
            max_grad[val, :] += X[num, :]
            max_grad[y[num],:] -= X[num, :]
        return W + C*max_grad

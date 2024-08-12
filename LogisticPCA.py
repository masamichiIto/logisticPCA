import numpy as np

class LogisticPCA:
    def __init__(self, ndim=2, nstarts = 10, max_iter = 300, tol=1e-4, trace=True, random_state=None):
        self.ndim = ndim
        self.nstarts = nstarts
        self.max_iter = max_iter
        self.tol = 1e-4
        self.trace = trace
        self.random_state = random_state
        self.losses = [np.Inf]
        self.min_loss = np.Inf

    def _logistic_func(self, X):
        return 1/(1+np.exp(-X))

    def _calc_loss(self, Q, A, B):
        return -np.sum(np.log(self._logistic_func(Q*(A@B.T))))

    def _first_deriv_logcdf(self, X):
        # calculate h(x) := -dlogF/dx
        # case: F(x) is a logistic cdf
        return self._logistic_func(X)-1


    def _AB_step(self, Z, W):
        lvec, sv, rvecT = np.linalg.svd(Z)
        #print(sv)
        A = lvec[:,:self.ndim]
        B = rvecT.T[:,:self.ndim]@np.diag(sv[:self.ndim])
        return A, B

    def _majorize_step(self, Q, A, B):
        W = 1/4 * np.ones((self.n, self.p))
        H = self._first_deriv_logcdf(Q*(A@B.T))
        Z = A@B.T - Q*H/W
        return W, H, Z

    def _initialize_AB_randomly(self, Q):
        Mat_temp = np.random.uniform(low=-1, high=1, size=self.n*self.p).reshape((self.n,self.p))
        lvec, sv, rvecT = np.linalg.svd(Mat_temp) # np.linalg.svdは A = UDV'の U, D, V'(Vは転置されたまま！！)返すことに注意.
        A = lvec[:, :self.ndim]
        B = rvecT.T[:, :self.ndim]@np.diag(sv[:self.ndim])
        return A, B

    def _initialize_AB_by_svd(self, Q):
        lvec, sv, rvecT = np.linalg.svd(Q)
        A = lvec[:, :self.ndim]
        B = rvecT.T[:, :self.ndim]@np.diag(sv[:self.ndim])
        return A, B

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n, self.p = X.shape
        Q = 2 * X - 1

        for nst in range(self.nstarts):
            if self.trace:
                print('--- {} ---'.format(nst+1))
            losses = [np.Inf]
            loss_min_temp = np.Inf
            # initialize A, B
            if nst == 0:
                A, B = self._initialize_AB_by_svd(Q)
            else:
                A, B = self._initialize_AB_randomly(Q)

            for itr in range(self.max_iter):
                # calculate parameters for majorizing loss function
                W, H, Z = self._majorize_step(Q, A, B)
                #print(W)
                #print(H)
                #print(Z)

                # update A, B
                A, B = self._AB_step(Z, W)

                loss_new = self._calc_loss(Q, A, B)
                if self.trace:
                    print('   {0}: {1:.4f}'.format(itr, loss_new))
                losses.append(loss_new)
                if 0 < (losses[itr] - losses[itr+1]) <= self.tol: # TODO収束判定の方法を再検討せよ．これだとlossのスケール次第では収束しない
                    loss_min_temp = losses[itr+1]
                    break
                elif itr+1 == self.max_iter:
                    print('warning: loss was not converged within iterations.')
                    loss_min_temp = losses[itr+1]

            if loss_min_temp < self.min_loss:
                self.A_hat = A
                self.B_hat = B
                self.losses = losses
                self.min_loss = loss_min_temp

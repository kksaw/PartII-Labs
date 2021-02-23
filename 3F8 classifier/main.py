# This is the auxiliary code for the 3F8 coursework. Some parts are missing and
# should be completed by the student. These are Marked with XXX

# We load the data

import numpy as np
import matplotlib.pyplot as plt

def main(plot=False):
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')
    
    # We randomly permute the data
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation, :]
    y = y[permutation]
    plot_data(X, y)
    
    # We split the data into train and test sets
    n_train = 800
    X_train = X[0:n_train, :]
    X_test = X[n_train:, :]
    y_train = y[0:n_train]
    y_test = y[n_train:]
    
    # We train the classifier
    alpha = 0.002   ##
    n_steps = 50  ##
    
    X_tilde_train = get_x_tilde(X_train)
    X_tilde_test = get_x_tilde(X_test)
    w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)
    
    # We plot the training and test log likelihoods
    plot_ll(ll_train)
    plot_ll(ll_test)
    plot_predictive_distribution(X, y, w)

    CM = confusion_matrix(X_tilde_test, y_test, w)
    '''
    [[0.71171171, 0.28828829],
     [0.26966292, 0.73033708]]
    '''
    
    
    l = 0.01 # XXX Width of the Gaussian basis funcction. To be completed by the student
    RBF_train = evaluate_basis_functions(l,X_train,X_train)
    RBF_test = evaluate_basis_functions(l,X_test,X_train)    
    RBF_tilde_train = get_x_tilde(RBF_train)
    RBF_tilde_test = get_x_tilde(RBF_test)
    
    alpha = 0.005
    n_steps = 1000
    w, ll_train, ll_test = fit_w(RBF_tilde_train, y_train, RBF_tilde_test, y_test, n_steps, alpha)
    
    plot_ll(ll_train, ll_test)
    plot_predictive_distribution(X, y, w, map_inputs=lambda x : evaluate_basis_functions(l, x, X_train))
    CM = confusion_matrix(RBF_tilde_test, y_test, w)
    
    j = np.random.permutation(X_train.shape[0])
    jk = X_train[j[:80]]
    BTS_train = evaluate_basis_functions(l,X_train,jk)
    BTS_test = evaluate_basis_functions(l,X_test,jk)    
    BTS_tilde_train = get_x_tilde(BTS_train)
    BTS_tilde_test = get_x_tilde(BTS_test)


def plot_data_internal(X, y, ms=5):
    x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
    y_min, y_max = X[ :, 1 ].min() - .5, X[ :, 1 ].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1', markersize=ms)
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2', markersize=ms)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy

def plot_data(X, y):
    xx, yy = plot_data_internal(X, y)
    plt.show()

def logistic(x): return 1.0 / (1.0 + np.exp(-x))

'''
## Predicts with logistic classifier ##
Input:
    X_tile: matrix of input features (with a constant 1 appended to the left) for which to make predictions
    w: vector of model parameters
Output: The predictions of the logistic classifier
'''
def predict(X_tilde, w): return logistic(np.dot(X_tilde, w))

'''
## Computes average loglikelihood of the logistic classifier on some data. ##
Input:
    X_tile: matrix of input features (with a constant 1 appended to the left) 
         for which to make predictions
    y: vector of binary output labels 
    w: vector of model parameters
Output: The average loglikelihood
'''
def compute_average_ll(X_tilde, y, w):
    output_prob = predict(X_tilde, w)
    return np.mean(y*np.log(output_prob) + (1-y)*np.log(1.0-output_prob))

def compute_individual_ll(X_tilde, y, w):
    output_prob = predict(X_tilde, w)
    return y*np.log(output_prob) + (1-y)*np.log(1.0-output_prob)

'''
## Expands a matrix of input features by adding a column equal to 1. ##
Input:
    X: matrix of input features.
Output: Matrix x_tilde with one additional constant column equal to 1 added.
'''

def get_x_tilde(X): return np.concatenate((np.ones((X.shape[ 0 ], 1 )), X), 1)


def fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
    '''
    ## Finds model parameters by optimising likelihood using gradient descent ##
    Input:
        X_tile_train: matrix of training input features (with a constant 1 appended to the left) 
        y_train: vector of training binary output labels 
        X_tile_test: matrix of test input features (with a constant 1 appended to the left) 
        y_test: vector of test binary output labels 
        alpha: step_size_parameter for the gradient based optimisation
        n_steps: the number of steps of gradient based optimisation
    Output: 
        1 - Vector of model parameters w 
        2 - Vector with average log-likelihood values obtained on the training set
        3 - Vector with average log-likelihood values obtained on the test set
    '''
    w = np.random.randn(X_tilde_train.shape[1])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)
    for i in range(n_steps):
        sigmoid_value = predict(X_tilde_train, w)
        w_new = w + alpha*np.dot(X_tilde_train.T, (y_train-sigmoid_value)) #!

        ll_train[i] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[i] = compute_average_ll(X_tilde_test, y_test, w)
        
        if i%10==0:
            print(ll_train[i], ll_test[i])
        
        w = w_new

    return w, ll_train, ll_test

'''
## Plot average log-likelihood returned by "fit_w" ##
Input:
    ll: vector with log-likelihood values
Output: Nothing
'''
def plot_ll(ll1, ll2):
    plt.figure()
    ax = plt.gca()
    ax.plot(np.arange(1, len(ll1) + 1), ll1, 'r-')

    if ll2!=[]:
        ax.plot(np.arange(1,len(ll2)+1),ll2, 'g-')
        plt.legend(['train','test'])
        
    plt.xlim(0, len(ll1)+2)
    plt.ylim(min(ll1) - 0.1, max(ll1) + 0.1)   
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()

'''
## Plot predictive probabilities of the logistic classifier ##
Input:
    X: 2d array with the input features for the data (without adding a constant column with ones at the beginning)
    y: 1d array with the class labels (0 or 1) for the data
    w: parameter vector
    map_inputs: function that expands the original 2D inputs using basis functions.
Output: Nothing.
'''
def plot_predictive_distribution(X, y, w, mode=0, var=0, map_inputs = lambda x : x,  ms=5):
    xx, yy = plot_data_internal(X, y, ms)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    
    if mode==0:
        Z = predict(X_tilde, w)
    elif mode==1:
        Z = compute_predictive_dist(X_tilde, w, var)
        
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()

def confusion_matrix(X,y,w):
    CM = np.zeros((2,2))
    ll_ind = compute_individual_ll(X,y,w)
    threshold = np.log(0.5)
    
    for j in [0,1]:
        check = [i for i, x in enumerate(y) if x == j]
        for i in check:
            if ll_ind[i] > threshold:
                CM[j][j] += 1
            else:
                CM[j][abs(j-1)] += 1
        CM[j] = CM[j]/len(check)
    return CM
        

'''
## Replace initial input features by evaluating Gaussian basis functions on a grid of points ##
Inputs:
    l: hyper-parameter for the width of the Gaussian basis functions
    Z: location of the Gaussian basis functions
    X: points at which to evaluate the basis functions
Output: Feature matrix with the evaluations of the Gaussian basis functions.
'''
def evaluate_basis_functions(l, X, Z):
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[0])
    ones_X = np.ones(X.shape[0])
    r2 = np.outer(X2, ones_Z) - (2*np.dot(X, Z.T)) + np.outer(ones_X, Z2)
    return np.exp(-0.5/l**2*r2)


''' FTR :OOOOOOO '''

'''LAPLACE APPROXIMATION
beta in notes is w
GOAL: approximate a Gaussian distribution for posterior so that you can solve
predictive distribution.
MAP
1. find mode of wMAP (scipy.optimize.fmin_l_bfgs_b) ok
2. evaluate Hessian at wMAP to get A
Laplace
1. Gaussian approximation q(w) = N(m0,S0)      m0=wMAP, S0=Sn
2. compute posterior distribution p(ynew|xnew,D)=sigma(kappa(sigma^2_a)mu_a) - need wMAP
3. approximate model evidence p(y|X)
'''
import scipy.optimize

def Lf(w, X, y, var):
    #w = w.reshape((w.shape[0],1)) 
    sigma = predict(X,w)
    ll_ind = np.dot(y.T, np.log(sigma)) + np.dot((1-y).T, np.log(1-sigma))
    ans = ll_ind - (0.5/var)*np.dot(w.T,w) - 0.5*X.shape[1]*np.log(2*np.pi*var)
    return -ans

def Lf1(w, X, y, var):
    #w = w.reshape((w.shape[0],1))
    sigma = predict(X,w)
    ans = np.dot(X.T, (y-sigma)) - (1/var)*w
    return -ans

def MAP(X1, y1, X2, y2, var, w0=None):
    if w0==None:
        w0 = np.random.normal(0,var,801)
    
    op = scipy.optimize.fmin_l_bfgs_b(Lf, w0, fprime=Lf1, args=[X1,y1,var], factr=100)
    print('MAP optimisation:' + str(op[2]['warnflag']))
    w = op[0]
    
    ll_train = compute_average_ll(X1, y1, w)
    ll_test = compute_average_ll(X2, y2, w)
    
    return w, ll_train, ll_test

'''gives same result as wMAP'''
def fit_wj(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha, var,mu=0.0):
    w = np.random.randn(X_tilde_train.shape[1])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)
    for i in range(n_steps):
        sigmoid_value = predict(X_tilde_train, w)
        prior = (w-mu)/var
        w_new = w + alpha*(np.dot(X_tilde_train.T, (y_train-sigmoid_value))-prior) #!

        ll_train[i] = compute_average_ll_BY(X_tilde_train, y_train, w, var)
        ll_test[i] = compute_average_ll_BY(X_tilde_test, y_test, w, var)
        
        if i%10==0:
            print(ll_train[i], ll_test[i])
        
        w = w_new

    return w, ll_train, ll_test

def compute_average_ll_BY(X_tilde, y, w, var):
    output_prob = compute_predictive_dist(X_tilde, w, var)
    return np.mean(y*np.log(output_prob) + (1-y)*np.log(1.0-output_prob))


def Hessian(X,w,var):
    sigma = predict(X,w)
    a = np.multiply(sigma,(1-sigma)).reshape((sigma.shape[0],1))
    b = np.multiply(a,X)
    c = np.dot(X.T,b)
    ans = c + (1/var)*np.identity(X.shape[1])
    return ans

def compute_Z(X, y, w, var): 
    H = Hessian(X,w,var)
    ev = np.linalg.eig(H)[0]
    
    if not all(i>0 for i in ev):
        print('H is not positive definite')
    
    log_det = np.sum(np.log(np.real(ev)))
    post = Lf(w, X, y, var)
    ans = -post + 0.5*X.shape[1]*np.log(2*np.pi) - 0.5*log_det    
    return ans

def compute_predictive_dist(X,w,var):
    H = Hessian(X,w,var)
    mua = np.dot(X,w) #200
    sigma2 = np.dot(np.dot(X,np.linalg.inv(H)),X.T) #200
    kappa = np.diag(1/np.sqrt(1+(np.pi*sigma2/8)))
    return logistic(np.multiply(kappa,mua))

def confusion_matrix_BY(pred,y):
    CM = np.zeros((2,2))
    threshold = 0.5
    for j in [0,1]:
        check = [i for i, x in enumerate(y) if x == j]
        for i in check:
            if pred[i] > threshold:
                CM[j][1] += 1
            else:
                CM[j][0] += 1
        CM[j] = CM[j]/len(check)
    return CM

def main_FTR(): 
    '''setup'''
    Xori = np.loadtxt('X.txt')
    yori = np.loadtxt('y.txt')

    permutation = np.random.permutation(Xori.shape[0])
    X = Xori[permutation, :]
    y = yori[permutation]

    n_train = 800
    X_train = X[0:n_train, :]
    X_test = X[n_train:, :]
    y_train = y[0:n_train]
    y_test = y[n_train:]

    '''hyperparameters'''
    l = 0.1     #width of RBFs
    var = 1.0   #sigma^2, variance of prior distribution p(w)
    
    '''input to model'''
    RBF_train = evaluate_basis_functions(l,X_train,X_train)
    RBF_test = evaluate_basis_functions(l,X_test,X_train)    
    RBF_tilde_train = get_x_tilde(RBF_train)
    RBF_tilde_test = get_x_tilde(RBF_test)
    
    alpha = 0.01
    n_steps = 1000
    
    '''MLE from lab'''
    wML, ll_trainML, ll_testML = fit_w(RBF_tilde_train, y_train, RBF_tilde_test, y_test, n_steps, alpha)
    plot_ll(ll_trainML, ll_testML)
    plot_predictive_distribution(X, y, wML, map_inputs=lambda x : evaluate_basis_functions(l, x, X_train))
    CMml = confusion_matrix(RBF_tilde_test, y_test, wML)
    print(CM)
        
    '''MAP'''
    wMAP, ll_trainMAP, ll_testMAP = MAP(RBF_tilde_train, y_train, RBF_tilde_test, y_test, var)
    plot_predictive_distribution(X, y, wMAP, map_inputs=lambda x : evaluate_basis_functions(l, x, X_train),ms=4)
    CMmap = confusion_matrix(RBF_tilde_test, y_test, wMAP)

    '''prediction using Laplace approx'''    
    wBY, ll_trainBY, ll_testBY = fit_wj(RBF_tilde_train, y_train, RBF_tilde_test, y_test, n_steps, alpha, var)
    pred = compute_predictive_dist(RBF_tilde_test,wBY,var)
    plot_predictive_distribution(X, y, wBY, var=var, mode=1, map_inputs=lambda x : evaluate_basis_functions(l, x, X_train),ms=4)
    CMby = confusion_matrix_BY(pred,y_test)
    
    '''calculate normalisation constant Z'''      
    l = np.linspace(0.3,0.8,10)
    var = np.linspace(0.2,1.2,10)
    Z,W = heatmap(l,var)
    plot_heatmap(Z, l, var)
    
    l = 0.5222
    var = 0.5333
    alpha = 0.001
    RBF_train = evaluate_basis_functions(l,X_train,X_train)
    RBF_test = evaluate_basis_functions(l,X_test,X_train)    
    RBF_tilde_train = get_x_tilde(RBF_train)
    RBF_tilde_test = get_x_tilde(RBF_test)
    
    wMAPn = MAP(RBF_tilde_train, y_train, RBF_tilde_test, y_test, var)[0]
    wBYn, ll_trainBYn, ll_testBYn = fit_wj(RBF_tilde_train, y_train, RBF_tilde_test, y_test, n_steps, alpha, var)
    predn = compute_predictive_dist(RBF_tilde_test,wBYn,var)
    plot_predictive_distribution(X, y, wBYn, var=var, mode=1, map_inputs=lambda x : evaluate_basis_functions(l, x, X_train),ms=4)
    CMbyn = confusion_matrix_BY(predn,y_test)



    
def heatmap(l, var, x=Xori, rbf=Xori, y=yori):
    '''  use Z from training or test?    plot l vs sigma  '''
    Z = np.zeros((10,10))
    W = np.zeros((10,10),dtype=object)
    for i in range(len(l)):
        X_tilde = get_x_tilde(evaluate_basis_functions(l[i],x,rbf))
        for j in range(len(var)):
            w0 = np.random.normal(0,var[j],X_tilde.shape[1])
            wMAP = scipy.optimize.fmin_l_bfgs_b(Lf, w0, fprime=Lf1, args=[X_tilde,y,var[j]], factr=100)
            print(str(wMAP[2]['warnflag']),end='')
            W[i,j] = wMAP[0]
            Z[i,j] = compute_Z(X_tilde, y, wMAP[0], var[j])
    
    return Z,W

def plot_heatmap(Z, l, var, interpol='gaussian'):
    fig, ax = plt.subplots()
    im = plt.imshow(Z, interpolation = interpol)
    plt.colorbar(im)
    a = np.where(Z == np.amax(Z))
    ax.plot(int(a[1]),int(a[0]),'r*',markersize=10)

    xlabel = [format(i,'.2f') for i in var]
    ylabel = [format(i,'.2f') for i in l]    
    ax.set_xticks(np.arange(0,Z.shape[1],1))
    ax.set_yticks(np.arange(0,Z.shape[0],1))
    ax.set_xticklabels(xlabel)
    ax.set_yticklabels(ylabel)
    ax.set_xlabel('$\sigma^2$')
    plt.ylabel('l')
    plt.title('Heatmap of model evidence')
    
    print('l: '+str(l[int(a[0])])+'    '+'var: '+str(var[int(a[1])]))

    
#def cdf(data):
#    s = np.sort(data)
#    p = 1.*np.arange(len(data))/(len(data)-1)    
#    return p
#
#    counts, bin_edges = np.histogram (data, bins=num_bins, normed=True)
#    cdf = np.cumsum (counts)
#    
#    scipy.stats.norm.cdf(x)


# -*- coding: utf-8 -*-
"""
Created on Sat May  9 07:40:40 2020

@author: khaik
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from script import CartPole as cp
from script import remap_angle, loss
from matplotlib.lines import Line2D
import sobol_seq
from scipy.optimize import minimize
from myfuncs import MidpointNormalize


def getv0(force=True,v1=10,v2=10,v3=np.pi,v4=16,v5=20):
    cl = np.random.uniform(-v1,v1)
    cv = np.random.uniform(-v2,v2)
    pa = np.random.uniform(-v3,v3)
    pv = np.random.uniform(-v4,v4)
    f = 0.0 if force==False else np.random.uniform(-v5,v5)
    return np.asarray([cl,cv,pa,pv,f])

def getvar(force=True,v1=10,v2=10,v3=np.pi,v4=16,v5=20,n=41):
    cll = np.linspace(-v1,v1,n)
    cvl = np.linspace(-v2,v2,n)
    pal = np.linspace(-v3,v3,n)
    pvl = np.linspace(-v4,v4,n)
    fl = np.zeros(n) if force==False else np.linspace(-v5,v5,n)
    xlabels =[np.linspace(-v1,v1,9),np.linspace(-v2,v2,9),
          np.asarray([r'-$\pi$', r'-$\frac{3\pi}{4}$', r'-$\frac{\pi}{2}$', r'-$\frac{\pi}{4}$', r'0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']),
          np.linspace(-v4,v4,9), np.linspace(-v5,v5,9)]
    return np.asarray([cll,cvl,pal,pvl,fl]), xlabels

titles = ['Cart location', 'Cart velocity', 'Pole angle', 'Pole velocity', 'External force']

index = {0:[0,1], 1:[0,2], 2:[0,3],
         3:[1,2], 4:[1,3], 5:[2,3]}

cart = cp()


'''SIMULATION'''
def Task11a(mode):
    '''time evolution'''
    if mode==0:     #simple oscillation
        a = [0,0,np.pi/2,0]
        figtitle = 'Simple oscillation time evolution'
    elif mode==1:   #full rotation
        a = [0,0,0,2]
        figtitle = 'Full rotation time evolution'

    cart.setState(a)
    cart.performAction()
    cart.sysvar[2] = [remap_angle(value) for value in cart.sysvar[2]]

    v = np.asarray(cart.sysvar)
    titles = ['Cart location', 'Cart velocity', 'Pole angle', 'Pole velocity']

    fig, axes = plt.subplots(2,2)
    plt.subplots_adjust(hspace=0.3,wspace=0.4,bottom=0.15)
    for ax, var, title in zip(axes.flatten(), v, titles):
        ax.plot(var)
        ax.set_xlabel('Time (s)')
        ax.set_xticks(np.linspace(0,cart.sim_steps,5))
        ax.set_xticklabels(np.linspace(0,cart.delta_time,5))
        ax.set_ylabel(title)
    fig.tight_layout()
    fig.suptitle(figtitle)
    fig.subplots_adjust(top=0.9)

def Task11b(mode,ndata=20,plot=None):
    '''phase profile    mode 0 - cvpv       mode 1 - papv '''
    cv = np.linspace(-15,15,ndata)
    pa = np.linspace(-np.pi, np.pi, ndata)
    pv = np.linspace(-15,15,ndata)
    nsteps = 33
    A,V = np.zeros((nsteps,ndata,ndata)), np.zeros((nsteps,ndata,ndata))

    if mode==0:
        X,Y = np.meshgrid(cv,pv)
    elif mode==1:
        X,Y = np.meshgrid(pa,pv)

    for i in range(ndata):
        for j in range(ndata):
            if mode==0:
                a = [0,cv[i],0,pv[j]]
            elif mode ==1:
                a = [0,0,pa[i],pv[j]]
            cart.setState(a)
            cart.performAction()
            for k in range(nsteps):
                A[k,i,j] = cart.sysvar[mode+1][k*15]
                V[k,i,j] = cart.sysvar[3][k*15]

    plt.figure()
    if mode==0:
        plt.xlabel('Cart velocity')
    elif mode==1:
        plt.xlabel('Pole angle')
    plt.ylabel('Pole velocity')

    if plot==None:
        for k in range(nsteps):
            plt.clf()
            plt.quiver(X,Y,A[k],V[k])
            plt.title(str(k*15))
            plt.pause(1)
    else:
        plt.quiver(X,Y,A[plot],V[plot])
        plt.title(str(plot*15))

def Task12a(mode, nrun=3, specific=None):
    '''(one step) change in state vars against var'''
    def statechange(mode,specific=specific):
        v0 = getv0(force=False) if specific==None else specific[k]
        data = []

        for w in var[mode]:
            v0[mode] = w
            cart.setState(v0[:4])
            cart.performAction(action=v0[4])
            vF = np.asarray(cart.sysvar)[:,-1]
            vDelta = vF-v0[:4]
            data.append(vDelta)

        print([x for i,x in enumerate(v0) if i!=mode])

        return np.asarray(data)

    var, xlabels = getvar(force=False)
    k=0
    while k<nrun:
        fig, axes = plt.subplots(2,2)
        ax = axes.flatten()
        plt.subplots_adjust(hspace=0.3,wspace=0.3,bottom=0.12)
        data = statechange(mode,specific=specific)

        for i in range(4):
            pos = data[:,i].copy()
            neg = data[:,i].copy()
            pos[pos <= 0] = np.nan
            neg[neg > 0] = np.nan

            ax[i].plot(pos,color='#f96a4a')
            ax[i].plot(neg,color='#7fbcff')
            ax[i].set_xlabel(titles[mode])
            ax[i].set_ylabel('Change in '+titles[i])
            ax[i].set_xticks([i*5 for i in np.arange(9)])
            ax[i].set_xticklabels(xlabels[mode])

        fig.suptitle('State changes')
        k+=1


def Task12b(mode, num=0, ndata=500, specific=True):
    '''TCF - change 2 var at a time'''
    cl = np.random.uniform(-10,10,ndata)
    cv = np.random.uniform(-10,10,ndata)
    pa = np.random.uniform(-np.pi,np.pi,ndata)
    pv = np.random.uniform(-16,16,ndata)
    random = [cl,cv,pa,pv]
    data = np.zeros((4,ndata))

    if specific:
        cl0,cv0,pa0,pv0 = num,num,num,num
    else:
        cl0 = cl[np.random.randint(0,ndata-1)]
        cv0 = cv[np.random.randint(0,ndata-1)]
        pa0 = pa[np.random.randint(0,ndata-1)]
        pv0 = pv[np.random.randint(0,ndata-1)]
    fixed = [cl0,cv0,pa0,pv0]

    for i in range(ndata):
        v0 = np.asarray([cl[i],cv[i],pa[i],pv[i]])
        v0[index[mode][0]] = fixed[index[mode][0]]
        v0[index[mode][1]] = fixed[index[mode][1]]

        cart.setState(v0[:4])
        cart.performAction(action=v0[4])
        vF = np.asarray(cart.sysvar)[:,-1]
        vDelta = vF-v0
        data[:,i]=vDelta

    print(v0)

    fig, axes = plt.subplots(2,2)
    triang = tri.Triangulation(random[index[mode][0]],random[index[mode][1]])
    for ax,d,title in zip(axes.flatten(),data,titles):
        im = ax.tricontourf(triang,d,cmap='RdBu_r')
        #ax.tricontour(triang,data, linewidths=0.5, colors='k')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(titles[index[mode][0]])
        ax.set_ylabel(titles[index[mode][1]])
        ax.set_title(title)
    fig.tight_layout(pad=2.0)

'''Task1.1 Dynamical Simulation'''
cart.sim_steps = 500 #50
cart.delta_time = 4.0   #0.2
Task11a(0), Task11a(1)
Task11b(1, plot=11)

'''Task 1.2 Changes of State'''
ndata = 500
cart.sim_steps = 50
cart.delta_time = 0.2

for i in range(4):
    Task12a(i)
for i in range(10):
    Task12b(i)


'''LINEAR MODEL'''
def gendata(ndata=500, delta_time=0.2, step_size=50,force=False):
    cart.delta_time = delta_time
    cart.sim_steps = 50
    cl = np.random.uniform(-10,10,ndata)
    pa = np.random.uniform(-np.pi,np.pi,ndata)
    cv = np.random.uniform(-10,10,ndata)
    pv = np.random.uniform(-16,16,ndata)
    F = [0.0 for i in range(ndata)] if force==False else np.random.uniform(-20,20,ndata)

    V0 = np.empty((0,5),dtype=float)
    Delta = np.empty((0,4),dtype=float)

    for i in range(ndata):
        v0 = [cl[i],cv[i],pa[i],pv[i],F[i]]
        cart.setState(v0[:4])
        cart.performAction(action=v0[4])

        vDelta = []
        for j in range(4):
            vDelta.append(np.asarray(cart.sysvar[j][-1]-v0[j]))
        V0 = np.append(V0, np.asarray(v0).reshape((1,5)),axis=0)
        Delta = np.append(Delta, np.asarray(vDelta).reshape((1,4)),axis=0)

    return V0, Delta

def sim(v0,model,nsteps=1,remap=True):
    state = np.asarray(v0[:4]).reshape((1,4))
    force = np.asarray(v0[4])
    oldstate = state
    n=0
    while n < nsteps:
        vector = np.append(oldstate,force)
        newstate = oldstate + np.matmul(vector,model)
        if remap:
            newstate[:,2] = remap_angle(newstate[:,2])
        state = np.append(state,newstate,axis=0)
        oldstate = newstate
        n+=1
    return state

def Task13(model, nsteps=1, remap=True, force=False):
    '''(one step) change in state var against var'''
    cm = plt.cm.tab20(np.arange(20))
    fig1, axes1 = plt.subplots(2,2)
    ax1 = axes1.flatten()
    plt.subplots_adjust(hspace=0.3,wspace=0.4,bottom=0.15)
    fig2, axes2 = plt.subplots(2,2)
    ax2 = axes2.flatten()
    plt.subplots_adjust(hspace=0.3,wspace=0.4,bottom=0.15)

    var, xlabels = getvar()
    v0 = getv0(force)
    title = v0.copy()
    for i in range(4):
        actual, pred = [], []
        for w in var[i]:
            v0[i] = w
            cart.setState(v0[:4])
            cart.performAction(action=v0[4])
            if remap:
                cart.sysvar[2] = [remap_angle(value) for value in cart.sysvar[2]]
            actual.append(np.asarray(cart.sysvar)[:,-1])
            pred.append(sim(v0,model,nsteps=nsteps,remap=remap)[-1])

        actual = np.asarray(actual).reshape((41,4))
        pred = np.asarray(pred).reshape((41,4))

        ax1[i].set_xlabel(titles[i]+' input')
        ax1[i].set_xticks([i*5 for i in np.arange(9)])
        ax1[i].set_xticklabels(xlabels[i])
        ax1[i].set_ylabel('Next step')
        for v in range(4):
            ax1[i].plot(actual[:,v],color=cm[v*2])
            ax1[i].plot(pred[:,v],color=cm[1+v*2])

        ax2[i].plot(actual[:,i],pred[:,i],color=cm[i*2])
        ax2[i].set_xlabel('Actual ' + titles[i])
        ax2[i].set_ylabel('Predicted ' + titles[i])

    custom_lines = [Line2D([0],[0],color=cm[c*2]) for c in range(4)]
    fig1.legend(custom_lines,titles,loc='lower center',ncol=4)
    fig1.suptitle(["%.2f"%title[t] for t in range(5)])


def Task14(model, delta_time=0.2, step_size=50, nsteps=10, remap=True, specific=None, force=False):
    '''time evolution'''
    cart.sim_steps = step_size*nsteps
    cart.delta_time = delta_time*nsteps

    fig, axes = plt.subplots(2,2)
    ax = axes.flatten()
    plt.subplots_adjust(hspace=0.3,wspace=0.4,bottom=0.15)

    xspace = int(nsteps/5)

    if specific == None:
        v0 = getv0(force)
    else:
        v0 = specific

    cart.setState(v0[:4])
    cart.performAction(action=v0[4])
    actual = np.asarray(cart.sysvar)
    if remap:
        actual[2] = np.asarray([remap_angle(value) for value in actual[2]])
    pred = sim(v0,model,nsteps=nsteps,remap=remap)
    nindex = len(actual[0])
    rindex = [index*step_size for index in range(int(np.ceil(nindex/step_size)))]

    for i in range(4):
        ax[i].plot(actual[i])
        ax[i].plot(rindex, pred[:,i])
        ax[i].set_xticks([n*step_size*xspace for n in np.arange(int(nsteps/xspace))])
        ax[i].set_xticklabels([n*xspace for n in np.arange(int((nsteps+1)/xspace))])
        ax[i].set_xlabel('Number of steps')
        ax[i].set_ylabel(titles[i])

    fig.legend(['Actual', 'Predicted'],loc='lower center', ncol=2)
    fig.suptitle('Actual vs predicted Time Evolution')
    print(v0)

'''Task 1.3 Linear Model'''
ndata = 500
sz, dt = 50, 0.2
var, xlabels = getvar()
x,y = gendata(ndata=500,delta_time=dt)
model = np.linalg.lstsq(x,y)[0]
Task13(model)


'''Task 1.4 Linear Model II'''
ndata = 500
cart = cp()
sz, dt = 50, 0.05
x,y = gendata(ndata=500, delta_time=dt, step_size=sz)
model = np.linalg.lstsq(x,y)[0]
Task14(model,delta_time=dt, step_size=sz, nsteps=50, specific=[0,3,np.pi/2,3])
Task14(model,delta_time=dt, step_size=sz, nsteps=50, specific=[0,3,np.pi/2,15])


'''NONLINEAR MODEL'''
def calc_K(X,Xprime,sigma):
    a = 0
    for j in range(5):
        if j==2:
            a += ((np.sin((X[j]-Xprime[j])/2))**2)/(2*sigma[j]**2)
        else:
            a += ((X[j]-Xprime[j])**2)/(2*sigma[j]**2)
    return np.exp(-a)

def calc_am(x,xprime,y,sigma,l=1e-6):
    Knm = np.zeros((len(x), len(xprime)))
    for i in range(len(x)):
        for k in range(len(xprime)):
            Knm[i,k] = calc_K(x[i],xprime[k],sigma)
    Kmn = Knm.T

    Kmm = np.zeros((len(xprime), len(xprime)))
    for i in range(len(xprime)):
        for k in range(len(xprime)):
            Kmm[i,k] = calc_K(xprime[i],xprime[k],sigma)

    A = np.matmul(Kmn,Knm)+l*Kmm
    b = np.matmul(Kmn,y)
    am = np.linalg.lstsq(A,b)[0]
    return am

def sim2(v0,model,nsteps=1,remap=False):
    sigma, xprime, am = model['sigma'], model['xprime'], model['am']
    state = v0[:4]
    force = v0[4]
    pred = [state]
    n=0
    while n<nsteps:
        curr = np.append(state,force)
        Knm = np.asarray([calc_K(curr,value,sigma) for value in xprime])
        y = np.matmul(Knm,am)
        state = state+y
        if remap:
            state[2] = remap_angle(state[2])
        pred.append(state)
        n+=1
    return np.asarray(pred)

def gen_bfloc(m,force=False):
    i0 = sobol_seq.i4_sobol_generate(5,m)   #quasi-random data location
    f = [40,20] if force else [0,0]
    i1 = np.diag([20,20,np.pi*2,32,f[0]])
    i2 = np.asarray([10,10,np.pi,16,f[1]])
    loc = np.matmul(i0,i1)-i2
    if force==False:
        loc[:,4]=0
    return loc

def check_error(model,n=1,remap=False,force=False,nv=10):
    VL = np.array([getv0(force=force) for i in range(nv)])
    error=0
    for v in VL:
        cart.setState(v[:4])
        cart.performAction(action=v[4])
        sysvar = np.asarray(cart.sysvar)[:,-1]
        if remap:
            sysvar[2] = remap_angle(sysvar[2])
        actual = sysvar
        pred = sim2(v,model,nsteps=n)[-1]
        error += np.mean((actual-pred)**2)
    return error

def check_params(remap=False, force=False, nv=10, dt=0.2):
    NL = np.array([500,1000,2000,4000])
    ML = np.array([10,20,40,80,160,320,640,1280])
    EM = np.zeros((len(NL),len(ML)))

    for i in range(len(NL)):
        x,y = gendata(ndata=NL[i],force=force,delta_time=dt)
        sigma = [np.std(x[:,i]) for i in range(5)]
        if not force:
            sigma[4]=1.0
        for j in range(len(ML)):
            if ML[j]>NL[i]:
                EM[i,j]=np.nan
                continue

            xprime = gen_bfloc(ML[j],force=force)
            am = calc_am(x,xprime,y,sigma,l=1e-6)
            model = {'sigma':sigma, 'am':am, 'xprime':xprime}
            EM[i,j] = check_error(model,force=force,nv=nv)/nv

    fig1, ax1 = plt.subplots(1)
    for p in range(4):
        ax1.plot(EM[p])
    ax1.set_xticks(np.linspace(0,8,9))
    ax1.set_xticklabels(ML)
    ax1.set_xlabel('# of basis functions')
    ax1.set_ylabel('MSE')
    ax1.legend(NL, title='# of datapoints')

    x,y = gendata(500,force=force,delta_time=dt)
    sigma = [np.std(x[:,i]) for i in range(5)]
    if not force:
        sigma[4]=1.0
    xprime = gen_bfloc(160,force=force)

    LL = np.logspace(-6,-1,10)
    EM = np.zeros(len(LL))

    for i in range(len(LL)):
        am = calc_am(x,xprime,y,sigma,l=LL[i])
        model = {'sigma':sigma, 'am':am, 'xprime':xprime}
        EM[i] = check_error(model,force=force,nv=nv)/nv

    fig2, ax2 = plt.subplots(1)
    ax2.plot(np.linspace(-6,-1,10),EM)
    ax2.set_xlabel(r'log($\lambda$)')
    ax2.set_ylabel('MSE')
    return

def NLModel(n=2000,m=160,l=1e-2,dt=0.2,force=True):
    x,y = gendata(ndata=n,force=force,delta_time=dt)
    sigma = [np.std(x[:,i]) for i in range(5)]
    if not force:
        sigma[4] = 1.0
    xprime = gen_bfloc(m,force=force)
    am = calc_am(x,xprime,y,sigma,l=l)
    return {'sigma':sigma, 'am':am, 'xprime':xprime}

def Task21a(n=2000, m=160,l=0.1,dt=0.2,
            specific=True,v=np.array([0,0,np.pi,0,0]),
            remap=False,force=False):
    '''one step ok'''
    cm = plt.cm.tab20(np.arange(20))
    fig1, axes1 = plt.subplots(2,2)
    ax1 = axes1.flatten()
    plt.subplots_adjust(hspace=0.3,wspace=0.4,bottom=0.15)
    fig2, axes2 = plt.subplots(2,2)
    ax2 = axes2.flatten()
    plt.subplots_adjust(hspace=0.3,wspace=0.4,bottom=0.15)

    model = NLModel(n=n,m=m,l=l,force=force,dt=dt)
    var,xlabels = getvar(force=True)
    v0 = getv0(force) if specific==False else np.array(v)
    title = v0.copy()

    for i in range(4):
        actual, pred = [], []
        for w in var[i]:
            v0[i]=w
            cart.setState(v0[:4])
            cart.performAction(action=v0[4])
            sysvar = np.asarray(cart.sysvar)[:,-1]
            if remap:
                sysvar[2] = remap_angle(sysvar[2])
            actual.append(sysvar)
            pred.append(sim2(v0,model)[-1])
        actual = np.asarray(actual)
        pred = np.asarray(pred).reshape((41,4))

        ax1[i].set_xlabel(titles[i]+' input')
        ax1[i].set_xticks([i*5 for i in np.arange(9)])
        ax1[i].set_xticklabels(xlabels[i])
        ax1[i].set_ylabel('Next step')
        for j in range(4):
            ax1[i].plot(actual[:,j],color=cm[j*2])
            ax1[i].plot(pred[:,j],color=cm[1+j*2])

        ax2[i].plot(actual[:,i],pred[:,i],color=cm[i*2])
        ax2[i].set_xlabel('Actual ' + titles[i])
        ax2[i].set_ylabel('Predicted ' + titles[i])

    custom_lines = [Line2D([0],[0],color=cm[c*2]) for c in range(4)]
    fig1.legend(custom_lines,titles,loc='lower center',ncol=4)
    fig1.suptitle(["%.2f"%title[t] for t in range(5)])


def Task21b(n=2000, m=160, delta_time=0.2, step_size=50, v=np.array([0,0,0.3,0,0]),
            l=0.1, nsteps=10, remap=True, specific=False, force=True):
    '''time evolution ok'''
    v0 = getv0(force) if not specific else v
    cart.sim_steps = step_size*nsteps
    cart.delta_time = delta_time*nsteps
    cart.setState(v0[:4])
    cart.performAction(action=v0[4])
    actual = np.asarray(cart.sysvar)

    model = NLModel(n=n,m=m,l=l,force=force,dt=delta_time)
    pred = np.asarray(sim2(v0,model,nsteps=nsteps,remap=remap))
    if remap:
        actual[2] = np.asarray([remap_angle(value) for value in actual[2]])
    aindex = len(actual[0])
    pindex = [index*step_size for index in range(int(np.ceil(aindex/step_size)))]

    cm = plt.cm.tab20(np.arange(20))
    fig, axes = plt.subplots(2,2)
    ax = axes.flatten()
    plt.subplots_adjust(hspace=0.3,wspace=0.4,bottom=0.15)
    xspace = int(nsteps/5)

    for i in range(4):
        ax[i].plot(actual[i])
        ax[i].plot(pindex, pred[:,i])
        ax[i].set_xticks([n*step_size*xspace for n in np.arange(int(nsteps/xspace))])
        ax[i].set_xticklabels([n*xspace for n in np.arange(int((nsteps+1)/xspace))])
        ax[i].set_xlabel('Number of steps')
        ax[i].set_ylabel(titles[i])

    fig.legend(['Actual', 'Predicted'],loc='lower center', ncol=2)
    fig.suptitle('Actual vs predicted Time Evolution')
    print(v0)
    return model

'''Task 2.1    sigma needs to be known, guessed or fitted'''
check_params(force=False)
Task21a(n=1000, m=160, l=1e-2, force=False)
Task21b(n=1000, m=160, l=1e-2, v=np.array([0,3,np.pi/2,3,0]), specific=True, force=False)
Task21b(n=1000, m=160, l=1e-2, v=np.array([0,3,np.pi/2,15,0]), specific=True, force=False)
#even with force, one step is ok for these params

check_params(force=True)
Task21a(n=2000, m=320, l=1e-2, force=True)
Task21b(n=2000, m=320, l=1e-2, v=np.array([0,3,np.pi/2,3,3]), specific=True, force=True)
Task21b(n=2000, m=320, l=1e-2, v=np.array([0,3,np.pi/2,15,3]), specific=True, force=True)

check_params(force=True, dt=0.05)
Task21a(n=4000, m=640, l=1e-2, dt=0.05, force=True)
model = Task21b(n=4000, m=640, l=1e-2, delta_time=0.05, nsteps=40, v=np.array([0,3,np.pi/2,3,3]), specific=True, force=True)
model = Task21b(n=4000, m=640, l=1e-2, delta_time=0.05, nsteps=40, v=np.array([0,3,np.pi/2,15,3]), specific=True, force=True)


'''LINEAR CONTROL'''
def Task23a(nsteps=10, mode=0, specific=False, ndata=500, v=[], force=False):
    cll = np.random.uniform(-100,100,ndata)
    cvl = np.random.uniform(-100,100,ndata)
    pal = np.random.uniform(-100,100,ndata)
    pvl = np.random.uniform(-100,100,ndata)
    var = [cll,cvl,pal,pvl]
    v0 = getv0(force=True,v1=0.5,v2=0.5,v3=0.2,v4=0.5,v5=0.5) if not specific else v
    print(v0, end= ' ')

    L = np.zeros(ndata, dtype=float)
    p = np.array([0.0, 0.0, 0.0, 0.0])

    v1,v2 = var[index[mode][0]], var[index[mode][1]]
    for i in range(ndata):
        p[index[mode][0]] = v1[i]
        p[index[mode][1]] = v2[i]

        state,a,n,l = v0[:4],v0[4],0,0
        while n<nsteps:
            a = np.dot(p,state.T)
            cart.setState(state)
            cart.performAction(action=a)
            l += np.mean([loss(s) for s in np.asarray(cart.sysvar).T])
            n+=1
        L[i] += l/nsteps
    print(np.min(L))

    fig = plt.figure()
    triang = tri.Triangulation(v1,v2)
    im = plt.tricontourf(triang,L,cmap='YlGnBu')
    plt.colorbar(im)
    plt.xlabel(titles[index[mode][0]])
    plt.ylabel(titles[index[mode][1]])
    #plt.title()
    fig.tight_layout(pad=2.0)
    return L

def LFunc(p,v0,nsteps=10):
    l,n = 0,0
    state,a = v0[:4], v0[4]
    while n<nsteps:
        a = np.dot(p,state.T)
        cart.setState(state)
        cart.performAction(action=a)
        cart.sysvar[2] = np.array([remap_angle(s) for s in cart.sysvar[2]])
        state = cart.sysvar[:,-1]
        l += np.mean([loss(s) for s in cart.sysvar.T])
        n+=1
    return l/nsteps

def Task23b():
    PAL = np.linspace(0,0.3,10)
    PVL = np.linspace(0,1,10)
    PM,L = np.zeros((len(PAL),len(PVL),4)), np.zeros((len(PAL),len(PVL)))
    for i in range(len(PAL)):
        for j in range(len(PVL)):
            v0 = np.array([0,0,PAL[i],PVL[j],0])
            p=np.array([0,0,100,100])
            m = minimize(LFunc,p,args=v0,method='nelder-mead')
            PM[i,j] = m['x']
            L[i,j] = m['fun']

    #PM[3,2] was problematic
    fig1, ax1 = plt.subplots(1)
    im = ax1.imshow(L,cmap='YlGnBu',interpolation='gaussian')
    plt.colorbar(im)
    ax1.set_ylabel('pole angle')
    ax1.set_xlabel('pole velocity')
    ax1.set_xticklabels([-0.2,0,0.2,0.4,0.6,0.8])
    ax1.set_yticklabels([0,0.05,0.1,0.15,0.2,0.25])
    ax1.set_title('Loss')

    fig2, ax2 = plt.subplots(1)
    im = ax2.imshow(PM[:,:,2],cmap='RdBu',interpolation='gaussian',norm=MidpointNormalize(midpoint=0))
    plt.colorbar(im)
    ax2.set_ylabel('pole angle')
    ax2.set_xlabel('pole velocity')
    ax2.set_xticklabels([-0.2,0,0.2,0.4,0.6,0.8])
    ax2.set_yticklabels([0,0.05,0.1,0.15,0.2,0.25])
    ax2.set_title('Pole angle parameter')

    fig3, ax3 = plt.subplots(1)
    im = ax3.imshow(PM[:,:,3],cmap='RdBu',interpolation='gaussian',norm=MidpointNormalize(midpoint=0))
    plt.colorbar(im)
    ax3.set_ylabel('pole angle')
    ax3.set_xlabel('pole velocity')
    ax3.set_xticklabels([-0.2,0,0.2,0.4,0.6,0.8])
    ax3.set_yticklabels([0,0.05,0.1,0.15,0.2,0.25])
    ax3.set_title('Pole velocity parameter')

def Task24(nsteps=10,nv=10,specific=False,v=[]):
    v0 = getv0(force=False,v1=0.3,v2=0.3,v3=0.25,v4=0.5,v5=0.01) if not specific else v

    pguess = np.array([0,0,100,100])
    m1 = minimize(LFunc,pguess,args=v0,method='nelder-mead')
    m2 = minimize(LFunc_model,pguess,args=(v0,model),method='nelder-mead')
    p1, p2 = m1['x'], m2['x']

    AM1,AM2,SM1,SM2 = [],[],[],[]
    state, a = v0[:4], v0[4]
    n=0
    cart.drawPlot()
    cart.delta_time=0.2
    while n<nsteps:
        a = np.dot(p1,state.T)
        AM1.append(a)
        cart.update=True
        cart.setState(state)
        cart.performAction(action=a)
        state = cart.sysvar[:,-1]
        SM1.append(cart.sysvar)
        cart.update=False
        n+=1
    SM1 = np.swapaxes(np.array(SM1),1,2).reshape(((cart.sim_steps+1)*nsteps,4))

    state, a = v0[:4], v0[4]
    n=0
    while n<nsteps:
        a = np.dot(p2,state.T)
        AM2.append(a)
        v = np.append(state,a)
        state = np.asarray(sim2(v,model,nsteps=4,remap=True))
        SM2.append(state)
        state = state[-1]
        n+=1
    SM2 = np.array(SM2).reshape((nsteps*5,4))

    fig, axes = plt.subplots(2,2)
    ax = axes.flatten()
    plt.subplots_adjust(hspace=0.3,wspace=0.4,bottom=0.15)
    xspace = int(nsteps/5)
    nindex = len(SM1)
    rindex = [i*8 for i in range(50)]

    for i in range(4):
        ax[i].plot(SM1[:,i])
        ax[i].plot(rindex, SM2[:,i])
        ax[i].set_xticks([n*step_size*xspace for n in np.arange(int(nsteps/xspace))])
        ax[i].set_xticklabels([n*xspace for n in np.arange(int((nsteps+1)/xspace))])
        ax[i].set_xlabel('Number of steps')
        ax[i].set_ylabel(titles[i])

    fig.legend(['Actual', 'Predicted'],loc='lower center', ncol=2)
    fig.suptitle('Actual vs predicted Time Evolution')
    print(v0)

def LFunc_model(p,v0,model,nsteps=10):
    l,n = 0,0
    state,a = v0[:4],v0[4]
    while n<nsteps:
        p[0], p[1] = 0,0
        a = np.dot(p,state.T)
        v = np.append(state,a)
        state = np.asarray(sim2(v,model,nsteps=4,remap=True))
        l += np.mean([loss(s) for s in state])
        state = state[-1]
        n+=1
    return l/nsteps




'''Task 2.3 Visualise loss function'''
for i in range(6):
    Task23a(mode=i,specific=True,v=np.array([-0.1,-0.1,0.2,0.1,0]))

Task23b()

'''Task 2.4 Linear control on model'''
model = NLModel(n=4000,m=640,l=1e-2,force=True,dt=0.05)
cart.delta_time=0.2
Task24(specific=True, v=np.array([-0.1,-0.1,0.2,0.1,0]))

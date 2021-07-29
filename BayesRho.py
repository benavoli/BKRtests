import numpy as np
#import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as snb
from numpy import matlib as mb
#import nystrom
import scipy as sp
from sklearn.metrics.pairwise import rbf_kernel
import editdistance


def rbf(X=[],Y=[],ell=1.0):    
    if (X==[]):
        return 'rbf'
    else:
        return rbf_kernel(X,Y,gamma=0.5/ell**2)
    
def stringkernel(K=[], G=[], ell=1.0):
     if (K==[]):
        return 'stringK'
     else:
        R=np.zeros((K.shape[0],G.shape[0]))
        for a in range(K.shape[0]):
            for b in range(G.shape[0]):
                #print(K[a],"\n",G[b])
                #aaa
                R[a,b] = sp.exp(-editdistance.eval(K[a,0] , G[b,0])**2/ell**2/2)
        return R


def estimateL(X):
    if X.shape[0]<10:
        l=1.0
    else:
        l = np.sqrt(0.5*np.median(sp.spatial.distance.pdist(X)**2))
        if l==0:
            l=1.0
    return l


def estimateLstring(X):
    if X.shape[0]<5:
        l=1.0
    else:
        R=[]
        for a in range(X.shape[0]):
            for b in range(a+1,X.shape[0]):
                #print(X[a] , X[b])
                R.append(editdistance.eval(X[a,0] , X[b,0])**2)
        #print(R)
        l = np.sqrt(0.5*np.median(np.array(R)))
        if l==0:
            l=1.0
    return l


def CatKern(X=[],Y=[],par=np.array([]),alpha=5,gamma=1):
    if (X==[]):
        return 'CatKern'
    else:
        sx=X.shape
        sy=Y.shape
        n=sx[0]
        #par=np.bincount(X[:,0].astype(int))
        #par=par/np.sum(par)
        X1=mb.repmat(X,1,sy[0])
        Y1=mb.repmat(np.transpose(Y),sx[0],1)
        row,col=np.where(X1==Y1)
        K=np.zeros((sx[0],sy[0]))
        K[row,col]=1 #np.exp(gamma*(1-par[X1[row,col].astype(int)]**alpha)**(1/alpha))
        return K

def kenels_params(Kerneltype,X,nystrom=True):
    par=np.array([])
    if Kerneltype=='rbf':
        par=estimateL(X)
    elif Kerneltype=='stringK':
        par=estimateLstring(X)
    elif Kerneltype=='CatKern':
        0
    return par


class BKR(): 
    def __init__(self,KernX,KernY):
        self.KernX=KernX
        self.KernY=KernY
        self.samples = []
    
            
    def sample(self,X,Y,parX=np.array([]),parY=np.array([]),nystrom=False, ncomponents = 30, random_state=42, nsamples=3000, WW=[], nullsamples=1):
        if WW!=[]:
            nsamples = WW.shape[0]
            
        sx=X.shape
        sy=Y.shape    
        if sx[0]!=sy[0]:
            raise ValueError('Length of X and Y is different!')
        n=sx[0]
        #np.random.seed(random_state)
        #low rank approximation
        if nystrom==True:
            n_components = min(n, ncomponents)            
            inds = np.random.permutation(n)
            basis_inds = inds[:n_components]
            basisX = X[basis_inds]
            basisY = Y[basis_inds]
            basisX_kernel = self.KernX(basisX, basisX, parX)
            basisY_kernel = self.KernY(basisY, basisY, parY)
            U, S, V = sp.linalg.svd(basisX_kernel)
            S = np.maximum(S, 1e-12)
            XX = np.dot(U / np.sqrt(S), V)
            # feature vector
            phix = np.dot(self.KernX(X, basisX, parX),XX.T)
            U, S, V = sp.linalg.svd(basisY_kernel)
            S = np.maximum(S, 1e-12)
            YY = np.dot(U / np.sqrt(S), V)
            # feature vector
            phiy = np.dot(self.KernY(Y, basisY, parY),YY.T)

        else:
            Kxx = self.KernX(X, X, parX)
            Kyy = self.KernY(Y, Y, parY)
        
        
        samples=np.zeros((nsamples,1), dtype=float)
        avg = 0.0
        countnull = 0
        for i in range(0,nsamples):
            if WW!=[]:
                W=WW[i:i+1,:]
            else:
                W =  np.random.dirichlet(np.ones(n), 1)  
            if nystrom==False:
                H = np.diag(W[0,:])-np.dot(W.T,W)
                den1 = np.trace(np.linalg.multi_dot([Kxx,H,Kxx,H]))
                den2 = np.trace(np.linalg.multi_dot([Kyy,H,Kyy,H]))

                #null mean
                
                #print(den2-np.trace(np.linalg.multi_dot([Kyy0,H,Kyy0,H])))
                #aa
                if den1*den2>0:
                    
                    samples[i,0] = np.trace(np.linalg.multi_dot([Kxx,H,Kyy,H]))/np.sqrt(den1*den2)
                    for ii in range(0,nullsamples):
                        ind=np.random.permutation(n)
                        Y0=Y[ind,:]
                        Kyy0 = self.KernY(Y0, Y0, parY)
                        #W =  np.random.dirichlet(np.ones(n), 1) 
                        #H = np.diag(W[0,:])-np.dot(W.T,W)
                        nulls = np.trace(np.linalg.multi_dot([Kxx,H,Kyy0,H]))/np.sqrt(den1*den2)
                        avg = avg + nulls
                        countnull = countnull + 1
                else:
                    samples[i,0] = 0.0
            else:
                #null mean
                #ind=np.random.permutation(n)
                #Y0=Y[ind,:]
                #phiy0 = np.dot(self.KernY(Y0, basisY, parY),YY.T)
                den1 = np.linalg.norm(np.linalg.multi_dot([(phix-W.dot(phix)).T,np.diag(W[0,:]),phix-W.dot(phix)]), 'fro')**2
                den2 = np.linalg.norm(np.linalg.multi_dot([(phiy-W.dot(phiy)).T,np.diag(W[0,:]),phiy-W.dot(phiy)]), 'fro')**2
                num  = np.linalg.norm(np.linalg.multi_dot([(phix-W.dot(phix)).T,np.diag(W[0,:]),phiy-W.dot(phiy)]), 'fro')**2
                
                if den1*den2>0:
                  
                    samples[i,0] =((num)/np.sqrt(den1*den2))
                    for ii in range(0,nullsamples):
                        ind=np.random.permutation(n)
                        Y0=Y[ind,:]
                        phiy0 = np.dot(self.KernY(Y0, basisY, parY),YY.T)
                        #W =  np.random.dirichlet(np.ones(n), 1) 
                        num0 = np.linalg.norm(np.linalg.multi_dot([(phix-W.dot(phix)).T,np.diag(W[0,:]),phiy0-W.dot(phiy0)]), 'fro')**2  
                        nulls =  num0/np.sqrt(den1*den2)
                        avg = avg + nulls
                        countnull = countnull + 1
                    avg = avg + nulls
                else:
                    samples[i,0] = 0.0
        #print("mean under exchangeability=",avg/countnull,countnull)
        #print(samples)
        return (samples-avg/countnull)/np.maximum(1,1-avg/countnull)


                
    def test(self,X,Y,nsamples=3000, probTh=0.8, ROPI=0.025, nystrom=False, ncomponents = 30, WW=[], nullsamples=1):
        sx=X.shape
        sy=Y.shape    
        self.ROPI = ROPI
        if sx[0]!=sy[0]:
            raise ValueError('Length of X and Y is different!')
        if sx[0]>3:
            #compute parameters for kernel
            typex = self.KernX()#return kernel name
            parX=kenels_params(typex,X,nystrom=nystrom)
            typey = self.KernY()#return kernel name
            parY=kenels_params(typey,Y,nystrom=nystrom)
            #print(parX,parY)
            self.samples=self.sample(X,Y,parX=parX,parY=parY,nsamples=nsamples,nystrom=nystrom,ncomponents=ncomponents, WW= WW, nullsamples=nullsamples)
            meanrho = np.mean(self.samples)
            ind=np.where(self.samples>ROPI)
            probRight=len(ind[0])/len(self.samples)
            probLeft =1-probRight
        else:
            meanrho=0
            probRight=0.5
            probLeft=0.5
            meanrho=ROPI
            
        
        d='Undecided'
        if probRight>probTh:
            d='Dependent'
        elif probLeft>probTh:
            d='Independent'
        return meanrho,[probLeft,probRight],d
    
    def plot(self,figsize=(12, 5), ROPI=[], xlabel='',filename=''):
        if ROPI==[]:
            ROPI = self.ROPI
        
        fig=plt.figure(figsize=figsize)      
        ax=snb.kdeplot(self.samples[:,0], shade=True) 
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False)

        plt.xticks(fontsize=10)    
        plt.yticks(fontsize=10) 
        plt.tick_params(axis="both", which="both", bottom=False, top=False,    
                        labelbottom=True, left=False, right=False, labelleft=True)
        plt.axvline(x=ROPI,color='orange')
        plt.legend(['ROPI','posterior'])
        #ax.text(rhoTh, 10, 'Rho', rotation='vertical', ha='right')
        #add label
        if xlabel!='':
            plt.xlabel(xlabel,fontsize=16)
        if filename!='':
            plt.savefig(filename)
        return fig
    
def jointdecision(BKRtable, ROPI=0.025, thresh=0.9, nperm = 50):
    #BKRtable is a matrix ncomparisons x mcrun
    #each element is a sample of dCor from the posterior 
    decision = np.zeros((BKRtable.shape[0],nperm))
    ndecision = np.zeros(nperm)
    for i in range(nperm):

        indp = np.random.permutation(BKRtable.shape[0])
        val = -np.max(np.vstack([np.sum(BKRtable[indp,:]>ROPI,axis=1),np.sum(BKRtable[indp,:]<ROPI,axis=1)]),axis=0)
        ind = np.argsort(val)
        Bsort = BKRtable[indp[ind],:]
        #print(Bsort)
        if np.sum(Bsort[0,:]>ROPI)/Bsort.shape[1]>thresh:
            decision[0,i]=1; #depend
            ndecision[i] = ndecision[i] + 1
        elif np.sum(Bsort[0,:]<ROPI)/Bsort.shape[1]>thresh:
            decision[0,i]=-1; #independ
            ndecision[i] = ndecision[i] + 1
        #import pdb; pdb.set_trace()
        for ii in range(1,Bsort.shape[0]):
            decisiontmp = np.hstack([decision[0:ii,i],1])
            decisiontmp1 = np.hstack([decision[0:ii,i],-1])
            if np.sum(np.prod(Bsort[0:ii+1,:]*decisiontmp.reshape(-1,1)>ROPI*decisiontmp.reshape(-1,1),axis=0))/Bsort.shape[1]>thresh:
                decision[ii,i]=1
                ndecision[i] = ndecision[i] + 1
            elif np.sum(np.prod(Bsort[0:ii+1,:]*decisiontmp1.reshape(-1,1)>ROPI*decisiontmp1.reshape(-1,1),axis=0))/Bsort.shape[1]>thresh:
                decision[ii,i]=-1
                ndecision[i] = ndecision[i] + 1
            else:
                break
        decision[:,i] = decision[np.argsort(indp[ind]),i]
    return decision[:,np.argmax(ndecision)]
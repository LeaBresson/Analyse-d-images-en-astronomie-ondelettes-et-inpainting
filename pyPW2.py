# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy as dp

#   Performs sparse deconvolution using the forward-backward algorithm
#
#   Input : b - N x M array (input image)
#           mask - N x M array (binary mask)
#           nmax - scalar (number of iterations)
#           J - scalar (number of starlet scales)
#           k_mad - scalar (value of the k-mad threshold)
#
#
#   Output : x - N x M array (estimated deconvolved image)


def Prox_Inp(b,mask,nmax=100,J=2,k_mad=3,tol=1e-4):

    import numpy as np
    from copy import deepcopy as dp

    x = dp(b)
    y = dp(x)

    L = 1
    
    tk = 1
    Go_On = 1
    it = 0

    while Go_On:
        
        it += 1

        # -- computation of the gradient
        
        g = - (b - mask*y)
            
        # -- gradient descent
    
        x_half = y - 1/L*g
    
        # -- thresholding / or applying mask
    
        c,w = Starlet_Forward(x=x_half,J=J)
    
        for s in range(0,J):
            thrd = k_mad*mad(w[:,:,s])
            w[:,:,s] = (w[:,:,s] - thrd*np.sign(w[:,:,s]))*(abs(w[:,:,s]) > thrd)
            
        xp = Starlet_Inverse(c=c,w=w)
        
        tkp = 0.5*(1 + np.sqrt(1 + 4*tk*tk))
        
        y = xp + (tk-1)/tkp * (xp - x)  
        
        d_iff = np.linalg.norm(xp - x)/(1e-12 + np.linalg.norm(xp))
        
        if d_iff < tol:
            Go_On = 0
        if it > nmax:
            Go_On = 0
                    
        x = dp(xp)        
        tk = dp(tkp)
                    
    return xp

################# DEFINES THE MEDIAN ABSOLUTE DEVIATION	
def mad(xin = 0):

	import numpy as np

	z = np.median(abs(xin - np.median(xin)))/0.6735
	
	return z

################# Useful codes
def length(x=0):

    l = np.max(np.shape(x))
    return l

################# 1D convolution	
def filter_1d(xin=0,h=0,boption=3):

    import numpy as np
    import scipy.linalg as lng
    import copy as cp    
    
    x = np.squeeze(cp.copy(xin));
    n = length(x);
    m = length(h);
    y = cp.copy(x);

    z = np.zeros(1,m);
    
    m2 = np.int(np.floor(m/2))

    for r in range(m2):
                
        if boption == 1: # --- zero padding
                        
            z = np.concatenate([np.zeros(m-r-m2-1),x[0:r+m2+1]],axis=0)
        
        if boption == 2: # --- periodicity
            
            z = np.concatenate([x[n-(m-(r+m2))+1:n],x[0:r+m2+1]],axis=0)
        
        if boption == 3: # --- mirror
            
            u = x[0:m-(r+m2)-1];
            u = u[::-1]
            z = np.concatenate([u,x[0:r+m2+1]],axis=0)
                                     
        y[r] = np.sum(z*h)
        

    a = np.arange(np.int(m2),np.int(n-m+m2),1)

    for r in a:
        
        y[r] = np.sum(h*x[r-m2:m+r-m2])
    

    a = np.arange(np.int(n-m+m2+1)-1,n,1)

    for r in a:
            
        if boption == 1: # --- zero padding
            
            z = np.concatenate([x[r-m2:n],np.zeros(m - (n-r) - m2)],axis=0)
        
        if boption == 2: # --- periodicity
            
            z = np.concatenate([x[r-m2:n],x[0:m - (n-r) - m2]],axis=0)
        
        if boption == 3: # --- mirror
                        
            u = x[n - (m - (n-r) - m2 -1)-1:n]
            u = u[::-1]
            z = np.concatenate([x[r-m2:n],u],axis=0)
                    
        y[r] = np.sum(z*h)
    	
    return y
 
################# 1D convolution with the "a trous" algorithm	
def Apply_H1(x=0,h=0,scale=1,boption=3):

	m = length(h)
	
	if scale > 1:
		p = (m-1)*np.power(2,(scale-1)) + 1
		g = np.zeros( p)
		z = np.linspace(0,m-1,m)*np.power(2,(scale-1))
		g[z.astype(int)] = h
	
	else:
		g = h
				
	y = filter_1d(x,g,boption)
	
	return y

################# 2D "a trous" algorithm
def Starlet_Forward(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):

	import copy as cp
 	
	nx = np.shape(x)
	c = np.zeros((nx[0],nx[1]),dtype=complex)
	w = np.zeros((nx[0],nx[1],J))

	c = cp.copy(x)
	cnew = cp.copy(x)
	
	for scale in range(J):
		
		for r in range(nx[0]):
			
			cnew[r,:] = Apply_H1(c[r,:],h,scale,boption)
			
		for r in range(nx[1]):
		
			cnew[:,r] = Apply_H1(cnew[:,r],h,scale,boption)
			
		w[:,:,scale] = c - cnew
		
		c = cp.copy(cnew);

	return c,w

def Starlet_Inverse(c,w):
    import numpy as np   
    
    return c + np.sum(w,axis=2)
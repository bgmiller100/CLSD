# -*- coding: utf-8 -*-
"""
demo.py

Run CLSD demo tests on sample images from the original LSDSAR package. 
Settings controlled in '__main__' function.

*NOTICE: This program is adapted from the MATLAB demo  code of LSDSAR:
*"LSDSAR, a Markovian a contrario framework for line segment detection in SAR images"
*by Chenguang Liu, Rémy Abergel, Yann Gousseau and Florence Tupin. 
*(Pattern Recognition, 2019).
*https://doi.org/10.1016/j.patcog.2019.107034
*Date of Modification: October 8, 2020.

Benjamin Miller
benjamin.g.miller@utexas.edu
"""

## IMPORT NECESSARY LIBRARIES
#from clsd import clsd
from scipy.io import loadmat
from scipy import ndimage
import numpy as np
import itertools
import faulthandler; faulthandler.enable()
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import clsdmpi 
import os
import cv2 as cv
import imageio

def hyperparam(I_full,a,d,t,folder,name):
    
    
    lines = np.load('%s/data_%s.npy'%(folder,name));
    sarimshow(I_full,lines,folder,name,name)

    dim=10
    l = np.size(np.asarray(lines))/dim
    lines = np.reshape(np.asarray(lines).T,(-1,dim),order='F')
    
    
    good=0 
    total=len(lines)
    const = 10.
    maxnfa = 0;
    nfaerr=0;
    for k in range(0,len(lines)):
            x1 = lines[k,0];
            y1 = lines[k,1];
            z1 = lines[k,2];
            x2 = lines[k,3];
            y2 = lines[k,4];
            z2 = lines[k,5];
            nfa = lines[k,9];
            if (nfa==101.00):
                nfaerr=nfaerr+1;
            #if((np.abs(x2-x1)>=const*np.abs(y2-y1))and(np.abs(x2-x1)>=const*np.abs(z2-z1))):
                #good = good+1;
            if (nfa>maxnfa):
                maxnfa=np.copy(nfa);
    mystr = "\n\ta: %.1f, d: %.1f, t: %.1f, tot: %d, good, %d, maxnfa: %.2f, 101err: %d\n"%(a,d,t,int(total),int(good),maxnfa,int(nfaerr))
    text_file=open("%s/DETECTIONS.txt"%folder,"a")
    text_file.write(mystr)
    text_file.close()




def plotgrad3(I,beta=4.,tol=1.):

    ## PLOT NORMALIZED TEST IMAGE (crop 3-sigma variace)
    v = np.copy(I)
    val_max = np.mean(v) + 3*np.std(v)
    v[v>val_max]=val_max
    m = np.min(v)
    v = (v-m)/(val_max-m)

    ## PLOT DETECTED LINES, TRACK MAX LENGTH
    fig=plt.figure(figsize=plt.figaspect(1./3.))
     
    ax = fig.add_subplot(1,3,1)
    zz= int (v.shape[2]/2)
    ax.imshow(v[:,:,zz].squeeze())

    ax = fig.add_subplot(1,3,2)
    yy = int (v.shape[1]/2)
    ax.imshow(v[:,yy,:].squeeze())


    ax = fig.add_subplot(1,3,3)
    xx = int (v.shape[0]/2)
    figshow = ax.imshow(v[xx,:,:].squeeze())
   
    cbar = fig.colorbar(figshow)
    plt.savefig('figs/3grad_beta%.2f_tol%.2f.png'%(beta,tol))

def makegrad2(I):
    print(I.shape)
    X = I.shape[0]
    Y = I.shape[1]
    modgrad = np.zeros((X,Y))
    azimg = np.zeros((X,Y))
    elimg = np.zeros((X,Y))
    align=np.zeros_like(azimg)
    beta = 4.
    I = np.clip(I,1,100000000);
    largeur=int(np.ceil(np.log(10)*beta))
    grad1x=np.zeros_like(I)
    grad1y=np.zeros_like(I)
    gradx=np.zeros_like(I)
    grady=np.zeros_like(I)

    for y in range(Y-1):
        for x in range(X-1):
            Mx=0;My=0;Mz=0;
            for h in range(-largeur,largeur):
                xh = min(max(x+h,0),X-1);
                yh = min(max(y+h,0),Y-1);
                coeff=np.exp(-abs(h)/beta)
                Mx=Mx+coeff*I[xh,y]
                My=My+coeff*I[x,yh]
            grad1x[x,y]=Mx
            grad1y[x,y]=My
    for y in range(Y-1):
        for x in range(X-1):
            Mxg=0;Myg=0;Mzg=0;
            Mxd=0;Myd=0;Mzd=0;
            for h in range(1,largeur):
                xh1 = max(x-h,0);
                yh1 = max(y-h,0);
                xh2 = min(x+h,X-1);
                yh2 = min(y+h,Y-1);
                coeff=np.exp(-abs(h)/beta)
                Mxg=Mx+coeff*grad1x[xh1,y]
                Myg=My+coeff*grad1y[x,yh1]
                Mxd=Mx+coeff*grad1x[xh2,y]
                Myd=My+coeff*grad1y[x,yh2]
            gradx[x,y]=np.log(Mxd/Mxg)
            grady[x,y]=np.log(Myd/Myg)
    for y in range(Y-1):
        for x in range(X-1):
            ay=grady[x,y]
            ax=gradx[x,y]
            an=np.sqrt(ax*ax + ay*ay)
            modgrad[x,y]=an
            if(an>0):
                azimg[x,y]=np.arctan2(ax,-ay)

            al = azimg[x,y]-np.pi/4
            if(al<0.):
                al = -al
            if(al>(np.pi*3/2)):
                al=al-np.pi/2.
                if(al<0.):
                    al=-al
            align[x,y]= al#<(np.pi/8)

    #plt.fig()
    plt.imshow(align)
    plt.savefig('figs/LLANGLE_GRADS/NEW/2dtest.png')
    return modgrad


def makegrad(Iin,beta=4.,tol=1.):
    I=np.copy(Iin)
    print(I.shape)
    print('imax ',I.max())
    X = I.shape[0]
    Y = I.shape[1]
    Z = I.shape[2]
    modgrad = np.zeros((X,Y,Z))
    azimg = np.zeros((X,Y,Z))
    elimg = np.zeros((X,Y,Z))
    #beta = 4.
    I = np.clip(I,1.,100000000.);
    
    largeur=int(np.ceil(np.log(10)*beta))
    #align = np.copy(azimg);
    for z in range(Z-1):
        #print(z)
        for y in range(Y-1):
            #print('\t y:',y)
            for x in range(X-1):
                Mgx=0.;Mdx=0.;Mgy=0.;Mdy=0.;Mgz=0.;Mdz=0.
                for hy in range(-largeur,largeur):
                    for hz in range(-largeur,largeur):
                        for hx in range(-largeur,largeur):
                            xx = min(max(x+hx,0),X-1);
                            yy = min(max(y+hy,0),Y-1);
                            zz = min(max(z+hz,0),Z-1);
                            coeff = np.exp(-(abs(hx)+abs(hy)+abs(hz))/beta)
                            #coeff = 1 + (-(abs(hx)+abs(hy)+abs(hz))/beta)/256.
                            #coeff=coeff*coeff*coeff*coeff*coeff*coeff*coeff*coeff;

                            c = coeff*I[xx,yy,zz]
                            if hx!=0:
                                if xx<=x:
                                    Mgx=Mgx+c
                                if xx>=x: 
                                    Mdx=Mdx+c
                            if hy!=0: 
                                if yy<=y:
                                    Mgy=Mgy+c
                                if yy>=y: 
                                    Mdy=Mdy+c
                            if hz!=0:
                                if zz<=z:
                                    Mgz=Mgz+c
                                if zz>=z: 
                                    Mdz=Mdz+c

                ax=np.log(Mdx/Mgx)
                ay=np.log(Mdy/Mgy)
                az=np.log(Mdz/Mgz)

                an=np.sqrt(ax*ax + ay*ay + az*az)
                modgrad[x,y,z]=an
                if(an>0):
                    azimg[x,y,z]=np.arctan2(ax,-ay)
                    elimg[x,y,z]=np.arccos(az/an)
                #align[x,y,z]=isalign(np.pi/2.,0,azimg[x,y,z],elimg[x,y,z],np.pi/(8.*tol))
    
    np.save('%s/azimg_%s'%(folder,name),azimg)
    np.save('%s/elimg_%s'%(folder,name),elimg)
    #np.save('%s/align_%s'%(folder,name),align)
    np.save('%s/mod_%s'%(folder,name),modgrad)
    return modgrad

def isalign(az1,el1,az2,el2,tol):
    x1 = np.array([np.cos(az1)*np.sin(el1),np.sin(az1)*np.sin(el1),np.cos(el1)])
    x2 = np.array([np.cos(az2)*np.sin(el2),np.sin(az2)*np.sin(el2),np.cos(el2)])
    diff = np.dot(x1,x2)
    return abs(diff)<np.cos(np.pi/2.-tol)
    #return np.cos(tol)<=diff

def aligncheck(folder,name,dir1=np.pi/2.,dir2=0., tol=1.):
    azimg = np.load('%s/azimg_%s.npy'%(folder,name))
    elimg = np.load('%s/elimg_%s.npy'%(folder,name))
    modimg= np.load('%s/mod_%s.npy'%(folder,name))
    align = np.zeros_like(azimg)
    X = azimg.shape[0]
    Y = azimg.shape[1]
    Z = azimg.shape[2]
    for z in range(Z-1):
        #print(z)
        for y in range(Y-1):
            #print('\t y:',y)
            for x in range(X-1):
                #align[x,y,z] = azimg[x,y,z]
                if modimg[x,y,z]>0:
                    #align[x,y,z]=elimg[x,y,z]
                    align[x,y,z]=isalign(dir1,dir2,azimg[x,y,z],elimg[x,y,z],np.pi/(8.*tol))   
    np.save('%s/align_%s'%(folder,name),align)

def reloader(datafolder='figs',savename='temp', lims = np.array([[-50,150],[-5,5],[-5,30]])):
    '''
    Helper alias for adjusting indices of conditioning surface plots,
    using saved output from prior runs.
    
    Input: datafolder : folder locaton of saved .npy files, e.g. 'figs', or 'figs/subfolder'
           savename   : name of .npy files to process 
    
    Output: Multiple figures into './figs'    
    '''

    ## RUN INTERPIM SHOW ON NEW LIMITS
    interpimshow(datafolder=datafolder,savename=savename,limits=limits,loaddata=1)

def interpimshow(lines=[],savename='temp', loaddata=0, limits=[],datafolder='figs'): 
    '''
    Plot conditioning surfaces and save data
    
    Input: lines :  (N*4,1) vector of conditioning suface points
           savename   : name for saving .png and .npy files 
           
    Output: Multiple figures, and npy data, into './figs'    
    '''

    if(loaddata==0): ## IF FIRST RUN
        ## CONVERT LINES LIST TO 2D ARRAY
        l = np.size(np.asarray(lines))/4
        lines = np.reshape(np.asarray(lines).T,(-1,4),order='F')
        print('num lines : %d'%l) 

        ## CONVERT OUTPUT DATA TO N-BY-4 ARRAY, GET LIMITS
        nfas = np.zeros((len(lines),))
        lengths,angles,widths = np.copy(nfas),np.copy(nfas),np.copy(nfas);
        for k in range(0,len(lines)):
            lengths[k] = lines[k,0]
            angles[k] = lines[k,1]
            widths[k] = lines[k,2]
            nfas[k] = lines[k,3]
        lims = np.array([[np.min(lengths),np.max(lengths)],[np.min(angles),np.max(angles)],[np.min(widths),np.max(widths)]])
    else: ## IF RELOADING 
        ## LOAD DATA
        loaddata = np.load("%s/data_%s.npy"%(datafolder,savename))
        lengths,angles,widths,nfas = loaddata[0],loaddata[1],loaddata[2],loaddata[3]
        lims = limits
    
    ## PLOT 3D SCATTER (COLOR AS 4TH DIMENSION FOR NFA VALUE)
    title='Test: %s'%savename
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    p=ax.scatter3D(lengths,angles,widths,c=nfas,cmap='viridis')
    ax.set_title(title); ax.set_xlabel('length'); ax.set_ylabel('angle'); ax.set_zlabel('width');
    cbar = fig.colorbar(p)
    cbar.ax.set_ylabel('conditional probability')
    plt.savefig('figs/3D_%s.png'%savename)
    
    ## PLOT 2D SCATTER PROJECTIONS (COLOR AS NFA VALUE)
    fig=plt.figure(figsize=plt.figaspect(1./3.))
    ax = fig.add_subplot(1,3,1)
    ax.set_xlim(lims[0,0],lims[0,1]); ax.set_ylim(lims[1,0],lims[1,1]);
    ax.scatter(lengths,angles,c=nfas,cmap='viridis')
    ax.set_title(title); ax.set_xlabel('lenth'); ax.set_ylabel('angle')
    ax = fig.add_subplot(1,3,2)
    ax.set_xlim(lims[0,0],lims[0,1]); ax.set_ylim(lims[2,0],lims[2,1]);
    ax.scatter(lengths,widths,c=nfas,cmap='viridis')
    ax.set_title(title); ax.set_xlabel('lenth'); ax.set_ylabel('width')
    ax = fig.add_subplot(1,3,3)
    ax.set_xlim(lims[1,0],lims[1,1]); ax.set_ylim(lims[2,0],lims[2,1]);
    p=ax.scatter(angles,widths,c=nfas,cmap='viridis')
    ax.set_title(title); ax.set_xlabel('angle'); ax.set_ylabel('width')
    cbar=fig.colorbar(p)
    cbar.ax.set_ylabel('conditional probability')
    plt.savefig('figs/2D_%s.png'%savename)

    ## PLOT 2D PROJECTIONS AS 3D TRIANGULAR MESH SURFACES
    fig=plt.figure(figsize=plt.figaspect(1./3.))
    ax = fig.add_subplot(1,3,1,projection='3d')
    ax.set_xlim(lims[0,0],lims[0,1]); ax.set_ylim(lims[1,0],lims[1,1]);ax.set_zlim(0,1);
    ax.plot_trisurf(lengths,angles,nfas,cmap='viridis')
    ax.set_title(title); ax.set_xlabel('lenth'); ax.set_ylabel('angle')
    ax = fig.add_subplot(1,3,2,projection='3d')
    ax.set_xlim(lims[0,0],lims[0,1]); ax.set_ylim(lims[2,0],lims[2,1]);ax.set_zlim(0,1);
    ax.plot_trisurf(lengths,widths,nfas,cmap='viridis')
    ax.set_title(title); ax.set_xlabel('lenth'); ax.set_ylabel('width')
    ax = fig.add_subplot(1,3,3,projection='3d')
    ax.set_xlim(lims[1,0],lims[1,1]); ax.set_ylim(lims[2,0],lims[2,1]);ax.set_zlim(0,1);
    p=ax.plot_trisurf(angles,widths,nfas,cmap='viridis')
    ax.set_title(title); ax.set_xlabel('angle'); ax.set_ylabel('width')
    cbar=fig.colorbar(p)
    cbar.ax.set_ylabel('conditional probability')
    plt.savefig('figs/surf_%s.png'%savename)
    
    ## SAVE DATA TABLES
    if(loaddata==0): 
        savedata = np.array([lengths,angles,widths,nfas])
        np.save('figs/data_%s'%savename,savedata)

def sarimshow(img,lines=[],folder='',title='',savename='temp'): 
    '''
    Plot detections over test image and save data
    
    Input: img   :  2D test image 
           lines :  (N*7,1) vector of detected line parameters
           savename   : name for saving .png and .npy files 
           
    Output: Detection plot, and npy data, into './figs'    
    '''
    ## CONVERT LINES LIST TO 2D ARRAY
    if img.ndim==2:
        a=7
    elif img.ndim==3:
        a=10
    else:
        print('ERROR: sarimshow not passed a 3D nor 2D image')
        return        
    l = np.size(np.asarray(lines))/a
    lines = np.reshape(np.asarray(lines).T,(-1,a),order='F')
    #Remove rows with nan or inf elements 
    lines = lines[np.isfinite(lines).any(axis=1)]
    
    print('num lines : %d'%l) 
    if(l>0):
        print('\n\nDETECTION\n\n')
        text_file=open("%s/DETECTIONS.txt"%(folder),"a")
        text_file.write("%s: %d\n"%(savename,l))
        text_file.close()
    
    ## PLOT NORMALIZED TEST IMAGE (crop 3-sigma variace)
    v = np.copy(img)
    thick = int(np.ceil(v.shape[0]/100))
    val_max = np.mean(v) + 3*np.std(v)
    v[v>val_max]=val_max
    m = np.min(v)
    v = (v-m)/(val_max-m)
    v = np.clip(v,np.mean(v)-3.*np.std(v),np.mean(v)+3.*np.std(v))

    v = np.array((v-v.min())/(v.max()-v.min())*255,np.uint8)
    v = np.stack((v,)*3,axis=-1)


    ## PLOT DETECTED LINES, TRACK MAX LENGTH
    max_len=0
    if img.ndim==2:
        if l>0:
            for k in range(0,len(lines)):
                cv.line(v,(int(lines[k,1]),int(lines[k,0])),(int(lines[k,3]),int(lines[k,2])),(255,0,0),thick)
                k_len=np.linalg.norm(lines[k,:2]-lines[k,2:4])
                if(k_len>max_len):
                    max_len=np.copy(k_len)
        imageio.imwrite('%s/%s_img.png'%(folder,savename),v)
    if img.ndim==3:
        
        zz= int (v.shape[2]/2)
        sub1 = np.copy(v[:,:,zz,:].squeeze())
        for k in range(0,len(lines)):
            cv.line(sub1,(int(lines[k,1]),int(lines[k,0])),(int(lines[k,4]),int(lines[k,3])),(255,0,0),thick)
            k_len=np.linalg.norm(lines[k,:3]-lines[k,3:6])
            if(k_len>max_len):
                max_len=np.copy(k_len)

        yy= int (v.shape[1]/2)
        sub2 = np.copy(v[:,yy,:,:].squeeze())
        for k in range(0,len(lines)):
            cv.line(sub2,(int(lines[k,2]),int(lines[k,0])),(int(lines[k,5]),int(lines[k,3])),(255,0,0),thick)

        xx= int (v.shape[0]/2)
        sub3 = np.copy(v[xx,:,:,:].squeeze())
        for k in range(0,len(lines)):
            cv.line(sub3,(int(lines[k,2]),int(lines[k,1])),(int(lines[k,5]),int(lines[k,4])),(255,0,0),thick)

        sub1=cv.copyMakeBorder(sub1,10,10,10,10,cv.BORDER_CONSTANT,value=(255,255,255))
        sub2=cv.copyMakeBorder(sub2,10,10,10,10,cv.BORDER_CONSTANT,value=(255,255,255))
        sub3=cv.copyMakeBorder(sub3,10,10,10,10,cv.BORDER_CONSTANT,value=(255,255,255))
        allsub = cv.hconcat((sub1,sub2,sub3))
        imageio.imwrite('%s/%s_img.png'%(folder,savename),allsub)

    print('max line length : %.2f\n'%max_len)
    
    ## SAVE DATA TABLES


def precompute_transition(alpha):
    '''
    Get default Markov transition kernel from LSDSA. 
    '''
    return 0.5874,0.0590,0.4465,0.0369,0.2811,0.0232

def precompute_transition3(alpha):
    '''
    Get default Markov transition kernel from LSDSA. 
    '''
    #return .4632,.2707,.3638,.1706,.2903,.1099
    return .2101,.1329,.1409,.0835,.0911,.0519


def main(I,I0,folder='figs',name='',a=4.,d=.4,t=1.,p=[0],getp=1):
    '''
    Run CLSD pipeline for a test and conditioning image pair .
    Reshape data and set input parameter vector.
    Decide between output functions.
    
    Input: I :  2D test image
           I0:  2D conditioning noise model image
           name   : name for saving .png and .npy files
           
    Output: Results from associated plotting functions

    Notes: I,I0 cannot be empty vectors.  To set an 'empty image',
           use a small vector, e.g. 2x2 identity, and the function
           will set the variables X,X0 appropriatly for clsd.
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
    ## SET INPUT DIRECTIONS FOR VERSIONING - CLSD CANNOT ACCEPT EMPTY IMAGES, BUT REQUIRES M/M0=0
    M,N,O,M0,N0,O0=0,0,0,0,0,0;
    print('Idim: %d, I0dim: %d'%(I.ndim,I0.ndim))
    if I.ndim==2:
        M,N=np.shape(I)
        O=0
    elif I.ndim==3:
        M,N,O=np.shape(I)
    else:
        print('ERROR; I is neither a 2D nor 3D image')
        return 
    print('M %d, N %d, O %d'%(M,N,O))
    if(M<=10):
        M=0; N=0;
       
    if I0.ndim==2:
        M0,N0=np.shape(I0)
        O0=0
    elif I0.ndim==3:
        M0,N0,O0=np.shape(I0)
    else:
        print('ERROR; I0 is neither a 2D nor 3D image')
        return 
    print('M0 %d, N0 %d, O0 %d'%(M0,N0,O0))
    if(M0<=10):
        M0=0; N0=0;

    ## FLATTEN IMAGES FOR USE IN C, COPY FOR PLOTTING 
    I_full = np.copy(I)
    I0_full= np.copy(I0)
    if (O==0):
        I = I.T.flatten().tolist();
    I0 = I0.T.flatten().tolist();
    if (O>0):
        templist = np.zeros((M*N*O,))
        for i in range(M):
            for j in range(N):
                for k in range(O):
                    templist[k+O*(i+M*j)] = I[i,j,k]
        I = templist.tolist()
    #print(I[:10])
    #I  = np.transpose(I,(2,1,0)).flatten().tolist()
    #I0 = np.transpose(I0,(2,1,0)).flatten().tolist()
    
    ## SET INPUT PARAMETER VECTOR 
    alpha=a#4.
    if O==0 and len(p)<6:
        p11,p01,p11_2,p01_2,p11_4,p01_4=precompute_transition(alpha)
    elif len(p)<6:
        p11,p01,p11_2,p01_2,p11_4,p01_4=precompute_transition3(alpha)
    else:
        p11=p[0];p01=p[1];p11_2=p[2];p01_2=p[3];p11_4=p[4];p01_4=p[5];
    eps=1/1.
    density=d #0.4
    angth=22.5
    sizenum=np.sqrt(M**2.+N**2.)*5.
    if sizenum>(10.**4):
        sizenum=10.**4
    if O>0:
        sizenum=min(10.**6.,np.sqrt(M**2.+N**2.+O**2.)*(5.))
        angth=22.5/t
    inputv=[alpha,eps,density,sizenum,angth,p11,p01,p11_2,p01_2,p11_4,p01_4]
    #for i in range(6,11):
    #    inputv[i]=inputv[i]/10
    ## RUN CLSD
    print('-------------------- TEST: %s --------------------'%name)
    if getp==1:
        markov=2
        lines = clsdmpi.clsdmpi(I,M,N,O,I0,M0,N0,O0,inputv,markov)
        return lines    
    else:
        markov=1
        if M0==0:
            markov=0
        print('inputv: ',inputv)
        lines = clsdmpi.clsdmpi(I,M,N,O,I0,M0,N0,O0,inputv,markov)
    #lines = [];
    ## PLOT RESULTS - EITHER CLSD/LSDSAR DETECTIONS OR CONDITIONING SURFACE

    np.save('%s/data_%s'%(folder,name),lines)
    if(M>0):
        #I_full = makegrad(I_full,beta=np.copy(alpha),tol=np.copy(angth))
        #sarimshow(I_full,lines,folder,'Test: %s'%name,name)
        pass
    else:
        interpimshow(lines=lines,savename=name)
    del I, I0, lines, alpha,density,angth, a,d,t,inputv,I_full,I0_full

if __name__=="__main__":
    
    emptyset = np.eye(2);
    ## BEGIN TIMING 
    starttime=time.time()
    numtests =1;
    '''
    Load images and run clsd pipelines, or plot previous results
    '''

    ## LOAD IMAGES 
    #High-noise line model
    #Test image - synthic model
    
    
    folder='figs/Z_9con'
    prename='MPIROT_9con'
    
    I3 = np.load('image/TACC/tacc_z_5wide_9con_500.npy')#[200:300,200:300,200:300]
    I3=(I3/I3.max())*4096

    I03 = np.load('image/TACC/tacc_noise_500.npy')#[200:300,200:300,200:300]
    I03=(I03/I03.max())*4096
    
    #I3 = np.zeros_like(I3)
    #I3[48:52,48:52,20:80] = 4500.
     
    emptyset = np.eye(2);
    p=[0.];

    I3 = ndimage.rotate(I3,30,reshape=False,axes=(0,1),mode='reflect')    
    I3 = ndimage.rotate(I3,30,reshape=False,axes=(1,2),mode='reflect')
    I3 = ndimage.rotate(I3,30,reshape=False,axes=(2,0),mode='reflect')
    #t1 a4, d3
    for t in (1.,):#(1.,2.,4.,):#(1.,2.,4.,):#np.arange(1.,6.,1.):
        for a in (4.,):#(2.,3.,4.,):#np.arange(1.,3.,.5):
            d = .4
            name = '%s_a%.1f_d%.1f_t%.1f'%(prename,a,d,t)            
            outv = main(np.copy(I3),np.copy(I03),folder,name,
                np.copy(a),np.copy(d),np.copy(t),p,getp=1)
            p=outv[5:]#p = [.5392,.2858,.3156,.1660,.1664,.0906]#outv[5:] 
            for d in (.4,):#(.3,.4,.5,):#np.arange(.1,.4,.1):    
                name = '%s_a%.1f_d%.1f_t%.1f'%(prename,a,d,t)
                main(np.copy(I3),np.copy(emptyset),folder,name,
                    np.copy(a),np.copy(d),np.copy(t),p,getp=0)
                #name = 'Z_a%.1f_d%.1f_t%.1f'%(a,d,t)
                #main(I32copy,I03copy,folder,name,np.copy(a),np.copy(d),np.copy(t))
    

    for t in (1.,):#(1.,2.,4.,):#np.arange(1.,6.,1.):
        for a in (4.,):#(2.,3.,4.,):#np.arange(1.,3.,.5):
            for d in (.4,):#(.3,.4,.5,):#np.arange(.1,.4,.1):
                name = '%s_a%.1f_d%.1f_t%.1f'%(prename,a,d,t)
                hyperparam(np.copy(I3),a,d,t,folder,name) 

                #modimg = np.load('%s/mod_%s.npy'%(folder,name))
                #alimg =  np.load('%s/align_%s.npy'%(folder,name))
                #lines = [];
                
                #lines =  np.load('%s/data_%s.npy'%(folder,name))
                #sarimshow(np.copy(I3),lines,folder,name,name)
                
                #sarimshow(alimg,lines,folder,'Align: %s'%name,'align_%s'%name)
   
    ## PLOT TIMING DATA 
    avgtime = (time.time()-starttime)/numtests
    print('AVERAGE RUNTIME: %.2f sec (%.2f min)\n\n'%(avgtime,avgtime/60.))

    

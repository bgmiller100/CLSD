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
from clsd import clsd
from scipy.io import loadmat
import numpy as np
import itertools
import faulthandler; faulthandler.enable()
import matplotlib
import matplotlib.pyplot as plt
import time

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

def sarimshow(img,lines=[],title='',savename='temp'): 
    '''
    Plot detections over test image and save data
    
    Input: img   :  2D test image 
           lines :  (N*7,1) vector of detected line parameters
           savename   : name for saving .png and .npy files 
           
    Output: Detection plot, and npy data, into './figs'    
    '''

    ## CONVERT LINES LIST TO 2D ARRAY 
    l = np.size(np.asarray(lines))/7
    lines = np.reshape(np.asarray(lines).T,(-1,7),order='F')
    print('num lines : %d'%l) 
    
    ## PLOT NORMALIZED TEST IMAGE
    v = np.copy(img)
    val_max = np.mean(v) + 3*np.std(v)
    v[v>val_max]=val_max
    m = np.min(v)
    v = (v-m)/(val_max-m)
    plt.figure()
    plt.imshow(v)

    ## PLOT DETECTED LINES, TRACK MAX LENGTH
    max_len=0
    if l>0:
        for k in range(0,len(lines)):
            plt.plot([lines[k,1],lines[k,3]],[lines[k,0],lines[k,2]],
                    linewidth=2,color='r')
            k_len=np.linalg.norm(lines[k,:2]-lines[k,2:4])
            if(k_len>max_len):
                max_len=np.copy(k_len)

    ## LABEL AND SAVE FIGURE 
    plt.title('%s, %d lines'%(title,l))
    plt.xlim(0,np.shape(v)[1]) ; plt.ylim(0,np.shape(v)[0])
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.savefig('figs/%s.png'%savename)
    print('max line length : %.2f\n'%max_len)
     
    ## SAVE DATA TABLES
    np.save('figs/data_%s'%savename,lines)


def precompute_transition(alpha):
    '''
    Get default Markov transition kernel from LSDSA. 
    '''
    return 0.5874,0.0590,0.4465,0.0369,0.2811,0.0232


def main(I,I0,name=''):
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

    ## SET INPUT DIRECTIONS FOR VERSIONING - CLSD CANNOT ACCEPT EMPTY IMAGES, BUT REQUIRES M/M0=0
    M,N=np.shape(I)
    if(M<=10):
        M=0; N=0;
    M0,N0=np.shape(I0)
    if(M0<=10):
        M0=0; N0=0;

    ## FLATTEN IMAGES FOR USE IN C, COPY FOR PLOTTING 
    I_full = np.copy(I)
    I0_full= np.copy(I0)
    I  = I.T.flatten().tolist()
    I0 = I0.T.flatten().tolist()
    
    ## SET INPUT PARAMETER VECTOR 
    alpha=4.
    p11,p01,p11_2,p01_2,p11_4,p01_4=precompute_transition(alpha)
    eps=1/1.
    density=0.4
    sizenum=np.sqrt(M**2.+N**2.)*5.
    if sizenum>(10.**4):
        sizenum=10.**4
    angth=22.5
    inputv=[alpha,eps,density,sizenum,angth,p11,p01,p11_2,p01_2,p11_4,p01_4]

    ## RUN CLSD
    print('-------------------- TEST: %S --------------------'%name)
    lines = clsd(I,M,N,I0,M0,N0,inputv)
    
    ## PLOT RESULTS - EITHER CLSD/LSDSAR DETECTIONS OR CONDITIONING SURFACE
    if(M>0):
        sarimshow(I_full,lines,'Test: %s'%name,name)
    else:
        interpimshow(lines=lines,savename=name)

if __name__=="__main__":
    '''
    Load images and run clsd pipelines, or plot previous results
    '''

    ## LOAD IMAGES 
    #High-noise line model
    I0d1 = loadmat('image/synthetic/1look/image_1_3.mat')
    I0d1 = (np.asarray(I0d1['image_1_3'],dtype=np.float64))
    #Low-noise line model
    I0d2 = loadmat('image/synthetic/3look/image_1_3.mat')
    I0d2 = (np.asarray(I0d2['image_1_3'],dtype=np.float64))
    #Pure noise 
    I0n = loadmat('image/purenoise/purebig.mat')
    I0n = (np.asarray(I0n['purebig'],dtype=np.float64))
    #Mixed noise model
    I0mix=np.copy(I0n)[:512,:512];
    I0mix[100:300,100:300] = np.copy(I0d2)[100:300,100:300]
    #Emtpy set for test versioning 
    emptyset = np.eye(2);
    #Test image - real data
    Ireal  = loadmat('image/real_SAR/saclay.mat')
    Ireal =  (np.asarray(I['saclay'],dtype=np.float64))
    #Test image - synthic model
    I = loadmat('image/synthetic/3look/image_1_9.mat')
    I = (np.asarray(I['image_1_9'],dtype=np.float64))
    
    ## BEGIN TIMING 
    starttime=time.time()
    numtests = 4;

    ## CLSD TESTS
    #main(I,np.copy(I0n )[:,:],'PureNoise')
    #main(I,np.copy(I0d1)[:512,:512],'Model1')
    #main(I,np.copy(I0d2)[:512,:512],'Model2')
    main(I,np.copy(I0mix)[:512,:512],'Mixed')
    
    ## LSDSAR TESTS
    #main(I,np.copy(emptyset),'TestImage')
    #main(np.copy(I0n )[:,:],np.copy(emptyset),'PureNoise')
    #main(np.copy(I0d1)[:512,:512],np.copy(emptyset),'Model1')
    #main(np.copy(I0d2)[:512,:512],np.copy(emptyset),'Model2')
    #main(np.copy(I0mix)[:512,:512],np.copy(emptyset),'Mixed')
    
    ## CONDITIONING SURFACES
    #main(emptyset,np.copy(I0n)[:,:],'PureNoise')
    #main(emptyset,np.copy(I0d1)[:512,:512],'LineModel1')
    #main(emptyset,np.copy(I0d2)[:512,:512],'LineModel2')
    #main(emptyset,np.copy(I0mix),'MixedModel')
    
    ## REPLOT SAVED DATA, FOR REFINING CONDITIONING SURFACE PLOT LIMTIS
    #datafolder='figs'
    #lims = np.array([[-50,150],[-5,5],[-5,30]])
    #reloader(datafolder=datafolder, savename='PureNoise',  lims=lims)
    #reloader(datafolder=datafolder, savename='LineModel1', lims=limgs)
    #reloader(datafolder=datafolder, savename='LineModel2', lims=lims)
    #reloader(datafolder=datafolder, savename='MixedModel', lims=lims) 

    ## PLOT TIMING DATA 
    avgtime = (time.time()-starttime)/numtests.
    print('AVERAGE RUNTIME: %.2f sec (%.2f min)\n\n'%(avgtime,avgtime/60.))

    

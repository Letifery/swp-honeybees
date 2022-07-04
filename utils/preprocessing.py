import time 
import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA
from random import randrange

class Preprocessing():
    def __init__(self, X):
        self.X = X
        
    def preprocess_data(self, mode:[(str, [float])]):
        '''
        Implements various preprocessing methods for the given data set, 
        such as cropping the images/slicing the data or normalisation.
        Parameters
        ----------
        X       : np.array(np.array([[float]])
                The dataset which contains all images
        mode    : [(str, parameters)]
                A list of operations and their respective parameters to be applied to X
                - cut: (start:int,end:int)
                    excludes all images of each image-subset which aren't in the specified 
                    interval (i.e. cut([0,1,2,3,4],[0,3]) -> [0,1,2])
                - slice: (n:int)
                    excludes all images, where their index isn't a multiple of n
                    (i.e. slice([0,1,2,3,4],[2]) -> [0,2,4])
                - PCA: (n:int, inc:bool, img_show:str or bool, var_cumu:n)
                    Uses PCA (or incremental PCA if inc=True) with n components on each image.
                    If var_cumu>0, a plot displaying the percentage of variance by each
                    principal component will appear (effectively explaining how many components
                    are needed to explain var_cumu% of variance)
        Returns
        ----------
        X       : The transformed dataset with all modification layers applied (retains the same shape)
        '''
    
        def cut_X(start, end):
            t = time.time()
            for i in range(len(self.X)):
                self.X[i] = self.X[i][start:end]
            print("[CUT] <%s,%s> time needed: %s seconds" % (start, end, time.time()-t))
    
        def slice_X_uniformly(n:int):
            t = time.time()
            for i in range(len(self.X)):
                self.X[i] = self.X[i][::n]
            print("[SLICE] <%s> time needed: %s seconds" % (n, time.time()-t))
    
        def PCA_X(n=None, inc:bool=False, show_img=False, var_cumu:float=95):
            t = time.time()
            if n is not None:
                svd = "full" if 0<n<1 else "auto"
                pca_X = IncrementalPCA(n_components=n, svd_solver = svd) if inc else PCA(n_components=n, svd_solver = svd)
            else:
                pca_X = IncrementalPCA() if inc else PCA()
            if type(show_img) == str or show_img == True:
                tmpr = randrange(len(self.X))
                idx = [0,0] if show_img=="fst" else [0,-1] if show_img=="lst" else [tmpr, randrange(len(X[tmpr]))]
                _, axes = plt.subplots(nrows=1, ncols=2)
                axes[0].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray)
            for i in range(len(self.X)):
                for k in range(len(self.X[i])):
                    self.X[i][k] = pca_X.fit_transform(self.X[i][k])
            print("[PCA] <%s> time needed: %s seconds" % (n, time.time()-t))
            if type(show_img) == str or show_img == True:
                axes[1].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray) 
                plt.show()
                
                k = np.cumsum(pca_X.explained_variance_ratio_)*100
                print("\nNumber of principal components needed to explain "+\
                    "%s percent of variance: %s" % (var_cumu, np.argmax(k>var_cumu)))
                plt.title('Cumulative Explained Variance explained by component')
                plt.ylabel('Cumulative Explained variance (%)')
                plt.xlabel('Principal components')
                plt.plot(k)
                plt.show()

        def scale_X(new_size:(int,int), show_img=False):
            t = time.time()
            if type(show_img) == str or show_img == True:
                tmpr = randrange(len(self.X))
                idx = [0,0] if show_img=="fst" else [0,-1] if show_img=="lst" else [tmpr, randrange(len(X[tmpr]))]
                _, axes = plt.subplots(nrows=1, ncols=2)
                axes[0].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray)
            for i in range(len(self.X)):
                for k,x in enumerate(self.X[i]):
                    self.X[i][k] = cv2.resize(self.X[i][k], (new_size[0], new_size[1]))
            if type(show_img) == str or show_img == True:
                axes[1].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray) 
                plt.show()
            print("[SCALE] <%s> time needed: %s seconds" % (new_size, time.time()-t))

        def norm_X(nrange=[0,1]):
            t = time.time()
            tmp = np.array([a for b in self.X for a in b])
            old_min, old_max = np.amin(tmp), np.amax(tmp)
            for i in range(len(self.X)):
                for k in range(len(self.X[i])):
                    self.X[i][k] = ((self.X[i][k] - old_min)/(old_max-old_min))*(nrange[1]-nrange[0])+nrange[0]
            print("[NORM] <%s> time needed: %s seconds" % (nrange, time.time()-t))
    
        def difference_X(step:int=2, dummy:bool=True):
            t = time.time()
            for i in range(len(self.X)):
                for k in range(len(self.X[i])):
                    try:
                        self.X[i][k] = self.X[i][k]-self.X[i+step][k+step]
                    except IndexError:
                        if dummy:
                            self.X[i][k] = np.zeros(self.X[i][k].shape)
                        else:
                            continue
            print("[DIFF] <%s,%s> time needed: %s seconds" % (step, dummy, time.time()-t))
    
        def rm_bg_X(mog:bool=True, show_img=False):
            t = time.time()
            bg_sub = cv2.createBackgroundSubtractorMOG2() if mog else cv2.createBackgroundSubtractorKNN()
        
            if type(show_img) == str or show_img == True:
                tmpr = randrange(len(self.X))
                idx = [0,0] if show_img=="fst" else [0,-1] if show_img=="lst" else [tmpr, randrange(len(X[tmpr]))]
                _, axes = plt.subplots(nrows=1, ncols=2)
                axes[0].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray)
            for i in range(len(self.X)):
                for k in range(len(self.X[i])):
                    self.X[i][k] = bg_sub.apply(self.X[i][k])
            print("[RMBG] <%s> time needed: %s seconds" % (mog, time.time()-t))
            if type(show_img) == str or show_img == True:
                axes[1].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray) 
                plt.show()

        def concatenate_X(mode:str="vtc", step:int=2, del_subsequent:bool=False, dummy:bool=True, show_img=False):
            t = time.time()
            conc_axis = 0 if mode=="vtc" else 1 if mode =="hzt" else -1
            if type(show_img) == str or show_img == True:
                tmpr = abs(randrange(len(self.X))-step if del_subsequent == True else 0)
                idx = [0,0] if show_img=="fst" else [0,-1] if show_img=="lst" else [tmpr, randrange(len(X[tmpr]))]
                _, axes = plt.subplots(nrows=1, ncols=2)
                axes[0].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray)
            for i in range(len(self.X)):
                for k in range(len(self.X[i])):    
                    for z in range(1, step):
                        try:
                            self.X[i][k] = np.concatenate((self.X[i][k], self.X[i+z][k+z]), axis=conc_axis)
                        except IndexError:
                            if not dummy:
                                continue
                            self.X[i][k] = np.zeros(self.X[i][k].shape)
                if del_subsequent:
                    self.X[i] = np.array(self.X[i])[np.arange(len(self.X[i]))%step==0]
            if type(show_img) == str or show_img == True:
                axes[1].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray) 
                plt.show()
            print("[CONC] <%s,%s,%s> time needed: %s seconds" % (mode, step, del_subsequent, time.time()-t))        
        
        def filter_X(ddepth=-1, kernel="sharp", show_img=False):
            t = time.time()
            if type(kernel) is str:
                kernel = {"sharp": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),}[kernel] 
            if type(show_img) == str or show_img == True:
                tmpr = randrange(len(self.X))
                idx = [0,0] if show_img=="fst" else [0,-1] if show_img=="lst" else [tmpr, randrange(len(X[tmpr]))]
                _, axes = plt.subplots(nrows=1, ncols=2)
                axes[0].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray)
            for i in range(len(self.X)):
                for k,x in enumerate(self.X[i]):
                    self.X[i][k] = cv2.filter2D(src=self.X[i][k], ddepth=ddepth, kernel=kernel)
            if type(show_img) == str or show_img == True:
                axes[1].imshow(self.X[idx[0]][idx[1]], cmap=plt.cm.gray) 
                plt.show()
            print("[FILTER] <%s> time needed: %s seconds" % (ddepth, time.time()-t))
            return self.X 

        func_dic = {"cut": cut_X, "slice": slice_X_uniformly, "scale": scale_X, "rmbg": rm_bg_X, "PCA": PCA_X, 
                    "filter": filter_X, "norm": norm_X, "diff": difference_X, "conc":concatenate_X}
        ta = time.time()
        for x in mode:
            func_dic[x[0]](*x[1])
        print("[ENDPP] time needed: %s seconds\n" % (time.time()-ta))
        return self.X

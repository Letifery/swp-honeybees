import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

from locale import normalize
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA
from random import randrange

class Preprocessing():
    def __init__(self, X):
        self.X = X


    def return_new_X(self, n = None):
        if self.X.ndim != 4:
            return np.zeros((np.shape(np.array(self.X))[0],
                   n if n != None else np.shape(np.array(self.X[0]))[0] ,
                   np.shape(np.array(self.X[0]))[1],
                   np.shape(np.array(self.X[0]))[2]),
                   dtype = 'uint8')
        else:
            return np.zeros((np.shape(self.X)), dtype = 'uint8')


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
                - slice_fixed_X_uniformly: (n:int)
                    excludes all images, where their index isn't a multiple of n
                    (i.e. slice([0,1,2,3,4],[2]) -> [0,2,4])
                - slice_X_uniformly: (n:int)
                    reduces the number of images to n, excludes every len(videoClip)/n image
                - slice_with_prob (n:int, kind_of_prob: str)
                    exluedes images based on a propablity distibution
                    kind_of_prob: start, end, gauss, endStart
                - crop_normalize (scale_counter: int, scale_denominator:int)
                    normalizes the data and crops the images with the scale_factor (scale_factor = scale_counter/ scale_denominator)
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

        def slice_fixed_X_uniformly(n:int):
            t = time.time()
            for i in range(len(self.X)):
                self.X[i] = self.X[i][::n]
            print("[SLICE] <%s> time needed: %s seconds" % (n, time.time()-t))

        def slice_X_uniformly(n):
            t = time.time()
            new_X = self.return_new_X(n)
            for i, sample in enumerate(self.X):
                if len(sample) > n:
                    n_uniformly_distributed_indexes = np.arange(start = 0, stop = len(sample)-1, step = len(sample)/n, dtype = int)
                    n_uniformly_distributed_indexes = np.append(n_uniformly_distributed_indexes, -1) if len(n_uniformly_distributed_indexes) < n else n_uniformly_distributed_indexes
                    new_X[i] = np.array(self.X[i])[n_uniformly_distributed_indexes]
                else:
                    raise ValueError("[ERROR] Given n = '%s' is higher than the number of images in a video clip." % n)
            self.X = new_X
            print("[SLICE] <%s> time needed: %s seconds" % (n, time.time()-t))

        def slice_with_prob(n, kind_of_prob = "gauss"):
            t = time.time()

            def prob_start_distribution(n):
                prob = np.array([1/index for index in range(1,n+1)])
                return prob/ prob.sum(axis=0,keepdims=1)

            def prob_end_distribution(n):
                prob = np.flip([1/index for index in range(1,n+1)])
                return prob/ prob.sum(axis=0,keepdims=1)

            def prob_gauss_distribution(n):
                prob = np.abs(np.random.normal(0.25, 0.1, size=(n, )))
                return prob/ prob.sum(axis=0,keepdims=1)

            def prob_start_end_distribution(n):
                    prob_end = [np.power(num, 2) for num in range(n//2)]
                    prob_start = np.flip(prob_end)
                    prob = np.concatenate([prob_start, prob_end]) if len(prob_start) + len(prob_end) == n else  np.concatenate([prob_start,  np.append(prob_end, prob_start[0])])
                    return prob/ prob.sum(axis=0,keepdims=1)

            kind_of_prob_dic = {"gauss": prob_gauss_distribution, "start": prob_start_distribution,
                                "end": prob_end_distribution, "endStart": prob_start_end_distribution}

            new_X = self.return_new_X(n)
            for i, videoClip in enumerate(self.X):
                new_X[i] = np.array(videoClip)[np.sort(np.random.choice(range(len(videoClip)), n, replace = False, p = kind_of_prob_dic[kind_of_prob](len(videoClip))))]
            self.X = new_X
            print("[SLICE RANDOM] <%s> time needed: %s seconds" % (n, time.time()-t))

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

        def crop_normalize(scale_counter = 8, scale_denominator = 55):
            def crop_images(images, crop):
                images = crop.augment_images(images)
                return images

            def floatify_image(img, normalize_to_float):
                img = img.astype(np.float32)
                if not np.issubdtype(img.dtype, np.floating):
                    assert img.max() > 1
                    img = img.astype(np.float32)
                else:
                    img = 255.0 * img
                img = normalize_to_float.augment_image(img)
                return img

            t = time.time()
            if np.shape(self.X) == np.shape(self.return_new_X()):

                scale_factor = scale_counter/ scale_denominator
                crop = iaa.Sequential([iaa.Resize(scale_factor),
                                       iaa.CenterCropToFixedSize(np.shape(self.X)[2], np.shape(self.X)[3]),])

                normalize_to_float = iaa.Sequential([iaa.Multiply(2.0 / 255.0),
                                                     iaa.Add(-1.0)])

                new_X = np.zeros((np.shape(self.X)[0],
                                  np.shape(self.X)[1],
                                  int(scale_factor * np.shape(self.X)[2]),
                                  int(scale_factor * np.shape(self.X)[3])),
                                  dtype = 'uint8')

                for i, videoClip in enumerate(self.X):
                    new_X[i] = np.array([floatify_image(img, normalize_to_float) for img in crop.augment_images(videoClip)])
                self.X = new_X
                print("[Crop and Normalize] time needed: %s seconds" % (time.time()-t))
            else:
                raise TypeError("[ERROR] Given X is not a numpy array of 4 dimensions." % n)

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

        func_dic = {"cut": cut_X, "slice": slice_fixed_X_uniformly, "slice_f": slice_X_uniformly, "slice_r": slice_with_prob, "scale": scale_X,
         "norm_crop": crop_normalize, "rmbg": rm_bg_X, "PCA": PCA_X, "filter": filter_X, "norm": norm_X, "diff": difference_X, "conc":concatenate_X}
        ta = time.time()
        for x in mode:
            func_dic[x[0]](*x[1])

        print("[END_PP] time needed: %s seconds\n" % (time.time()-ta))
        return self.X

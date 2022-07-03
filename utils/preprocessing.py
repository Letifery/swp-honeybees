class Preprocessing():
    def preprocess_data(X, mode:[(str, [float])]):
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
    
    def cut_X(X, start, end):
        t = time.time()
        for i in range(len(X)):
            X[i] = X[i][start:end]
        print("[CUT] <%s,%s> time needed: %s seconds" % (start, end, time.time()-t))
        return X
    
    def slice_X_uniformly(X, n:int):
        t = time.time()
        for i in range(len(X)):
            X[i] = X[i][::n]
        print("[SLICE] <%s> time needed: %s seconds" % (n, time.time()-t))
        return X 
    
    def PCA_X(X, n:int, inc:bool=True, show_img=False, var_cumu:float=95):
        t = time.time()
        pca_X = IncrementalPCA(n_components=n) if inc else PCA(n_components=n)
        if type(show_img) == str or True:
            tmpr = randrange(len(X))
            idx = [0,0] if show_img=="fst" else [0,-1] if show_img=="lst" else [tmpr, randrange(len(X[tmpr]))]
            _, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].imshow(X[idx[0]][idx[1]], cmap=plt.cm.gray)
        for i in range(len(X)):
            for k in range(len(X[i])):
                X[i][k] = pca_X.fit_transform(X[i][k])
        print("[PCA] <%s> time needed: %s seconds" % (n, time.time()-t))
        if type(show_img) == str or True:
            axes[1].imshow(X[idx[0]][idx[1]], cmap=plt.cm.gray) 
            plt.show()
        if var_cumu:
            k = np.cumsum(pca_X.explained_variance_ratio_)*100
            print("\nNumber of principal components needed to explain "+\
                "%s percent of variance: %s" % (var_cumu, np.argmax(k>var_cumu)))
            plt.title('Cumulative Explained Variance explained by component')
            plt.ylabel('Cumulative Explained variance (%)')
            plt.xlabel('Principal components')
            plt.plot(k)
            plt.show()
        return(X)
    
    p_X = X.copy()
    func_dic = {"cut": cut_X, "slice": slice_X_uniformly, "PCA": PCA_X}
    for x in mode:
        p_X = func_dic[x[0]](p_X, *x[1])
    return p_X

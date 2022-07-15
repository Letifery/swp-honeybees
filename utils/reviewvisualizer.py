import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ReviewVisualizer():
    def __init__(self, paths):
        self.dfs = [None]*len(paths)
        for i, path in enumerate(paths):
            self.dfs[i] = pd.read_csv(path, on_bad_lines='skip', usecols=(range(10)), header=None)      #Load df/relevant columns
            self.dfs[i] = self.dfs[i].rename(columns=self.dfs[i].iloc[0]).drop(self.dfs[i].index[0])    #Set first row as header
            self.dfs[i].columns = self.dfs[i].columns.str.lstrip(" ")                                   #Remove whitespaces
            self.dfs[i].drop(self.dfs[i].loc[self.dfs[i].CPU == " ERROR"].index, inplace=True)          #Remove rows with ERROR cells
            self.dfs[i] = self.dfs[i].loc[:, self.dfs[i].columns!="ID"].applymap(lambda x: float(x[1:]))#Clean and floatify cells
        
    def visualize_data(self, dfs, mp_dim, typemask, x, y, titles, xlabels, ylabels, legend_list):
        _, axes = plt.subplots(nrows=mp_dim[0], ncols=mp_dim[1])
        nflag = x is None 
        for df_id in dfs:
            if nflag: x = range(len(self.dfs[df_id].index))
            for i in range(mp_dim[0]):
                for k in range(mp_dim[1]):
                    z = k if mp_dim[0]==1 else i if mp_dim[1]==1 else (i,k)
                    if typemask[k+i*mp_dim[1]] == "line": 
                        axes[z].plot(x, self.dfs[df_id][y[k+i*mp_dim[1]]])
                    elif typemask[k+i*mp_dim[1]] == "scatter":
                        axes[z].scatter(x, self.dfs[df_id][y[k+i*mp_dim[1]]])
                    axes[z].set_xlabel(xlabels[k+i*mp_dim[1]])
                    axes[z].set_ylabel(ylabels[k+i*mp_dim[1]])
                    axes[z].set_title(titles[k+i*mp_dim[1]])
        for df_id in dfs:
            for i in range(mp_dim[0]):
                for k in range(mp_dim[1]):
                    axes[k if mp_dim[0]==1 else i if mp_dim[1]==1 else (i,k)].legend(legend_list)
        plt.show()

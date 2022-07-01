import numpy as np
import pandas as pd
import pickle
import pathlib
import json
import os
from io import StringIO
from tifffile import imread
import zipfile

from PIL import Image
from contextlib import contextmanager
from urllib import request, error

class DataLoader():
    @contextmanager
    def set_posix_windows(self):
        try:
            tmp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            yield
        finally:
            pathlib.PosixPath = tmp 
            
    def get_pickle(self, pointers:[str]):
        #pointers = URL/Path pointing to .pickle
        for pointer in pointers:
            try:                                                #Tries to grab the .pickle from an URL
                yield(pickle.load(request.urlopen(pointer)))
            except NotImplementedError:                         #PosixPath/WindowsPath compability (URL)
                with self.set_posix_windows():
                    yield(pickle.load(request.urlopen(pointer)))
            except error.URLError:                              #Checks if it is a dir.path if URL-grab failed
                try:
                    yield(pd.read_pickle(pointer))
                except NotImplementedError:                     #PosixPath/WindowsPath compability (Path)
                    with self.set_posix_windows():
                        yield(pd.read_pickle(pointer))
            except:
                raise FileNotFoundError("[ERROR] Given path/URL '%s' doesn't point to a .pickle" % path)
                
    def get_data(self, paths:[str], modes:[str] = ["unpackpng", "incpath"]):
        #unpackpng -> Png to np.array([int])
        #incpath -> Adds a path to each data entry
        for root in paths:
            for path, _, files in os.walk(root):
                data, jsondata = None, None
                for name in files:
                    if pathlib.Path(name).suffix == ".json":
                        try:
                            with open(os.path.join(path, name)) as file:
                                jsondata = json.load(file)
                        except:
                            raise SystemExit("Something went horribly wrong with the .json data at: %s" % path)
                    else:
                        try:
                            with zipfile.ZipFile(os.path.join(path, name), "r") as archive:
                                data = [None]*len(archive.namelist())
                                for i, img in enumerate(archive.namelist()):
                                    with archive.open(img, "r") as fd:
                                        with Image.open(fd) as f:
                                            data[i] = np.array(f)
                        except Exception as e:
                            print("[ERROR]", str(e))
                            raise SystemExit("Something went horribly wrong with the .zip archive at: %s" % path)
                yield([data, jsondata, path])
#Beispiele:
'''
path_angles = ["https://box.fu-berlin.de/s/nyweXr2oQzfHmHp/download?path=%2F&files=ground_truth_wdd_angles.pickle"]
path_angles = [r"I:\tmp_swp\ground_truth_wdd_angles.pickle"]
dl = DataLoader()
print(list(dl.get_pickle(path_angles)))
print(list(dl.get_data([r"I:\tmp_swp\wdd_ground_truth"])))
'''
path_angles = ["https://box.fu-berlin.de/s/nyweXr2oQzfHmHp/download?path=%2F&files=ground_truth_wdd_angles.pickle"]
dl = DataLoader()
print(list(dl.get_pickle(path_angles)))
print(list(dl.get_data([r"I:\tmp_swp\wdd_ground_truth"]))[0])

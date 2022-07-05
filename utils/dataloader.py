import numpy as np
import pandas as pd
import pickle
import pathlib
import json
import os
import zipfile, requests, io
import time

from tifffile import imread
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
            try:                                                    #Tries to grab the .pickle from an URL
                yield(pickle.load(request.urlopen(pointer)))
            except NotImplementedError:                             #PosixPath/WindowsPath compability (URL)
                with self.set_posix_windows():
                    yield(pickle.load(request.urlopen(pointer)))
            except (error.URLError, ValueError):                    #Checks if it is a dir.path if URL-grab failed
                try:
                    yield(pd.read_pickle(pointer))
                except NotImplementedError:                         #PosixPath/WindowsPath compability (Path)
                    with self.set_posix_windows():
                        yield(pd.read_pickle(pointer))
            except:
                raise FileNotFoundError("[ERROR] Given path/URL '%s' doesn't point to a .pickle" % pointer)
                
    def get_data(self, paths:[str]):
        #unpackpng -> Png to np.array([int])
        #incpath -> Adds a path to each data entry
        fpaths = []
        for path in paths:                                          
            try:
                with zipfile.ZipFile(io.BytesIO(requests.get(path).content)) as archive:
                    sdir = [f.path for f in os.scandir(os.getcwd()) if f.is_dir()]
                    archive.extractall(os.getcwd())
                    fpaths += [x for x in [f.path for f in os.scandir(os.getcwd()) if f.is_dir()] if x not in sdir]
            except:
                fpaths += [path]
        for root in fpaths:
            for path, _, files in os.walk(root):
                if files == []: continue
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

    def get_json(self, pointers):
        for pointer in pointers:
            with open(pointer) as f:
                yield(json.load(f))

import os
import platform
import warnings

from pathlib import Path
from datetime import datetime
from traceback import format_exc

class Logger:
    def __init__(self, filename=None, delimiter=","):
        if filename is not None:
            self.filename = filename 
        else: 
            self.filename = "%s-%s.log" % (Path(__file__).stem, datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        self.delimiter = delimiter
 
    def log_data(self, data:[[]], folder="datalogs"):
        sl = "/" if platform.system() == "Linux" else "\\."[0]
        if not os.path.exists(os.path.join(Path(__file__).parent.parent.resolve(),"logs"+sl+"%s" % folder)):
            os.makedirs(os.path.join(Path(__file__).parent.parent.resolve(), "logs"+sl+"%s" % folder))
        try:
            with open(os.path.join(Path(__file__).parent.parent.resolve(), "logs"+sl+"%s" % folder + sl + "%s" % self.filename), "a") as log:
                for row in data:
                    log.write((((("%s"+"%s " % self.delimiter)*len(row))[:-2]) % (tuple(row)))+"\n")
            return(1)
        except Exception as e:
            print(format_exc())
            warnings.warn("\n[WARNING] Couldn't create or write in logfile %s. Will skip logging in the future" % self.filename)
            return (-1)

import os

from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, filename=None):
        if filename is not None:
            self.filename = filename 
        else: 
            self.filename = "%s-%s.log" % (Path(__file__).stem, datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
 
    def log_data(self, data:[[]], folder="datalogs"):
        if not os.path.exists(os.path.join(Path(__file__).parent.resolve(),"logs\%s" % folder)):
            os.makedirs(os.path.join(Path(__file__).parent.resolve(), "logs\%s" % folder))
        try:
            with open(os.path.join(Path(__file__).parent.resolve(), "logs\%s\%s" % (folder, self.filename)), "a") as log:
                for row in data:
                    log.write(((("%s, "*len(row))[:-2]) % tuple(row))+"\n")
            return(1)
        except:
            warn("\n[WARNING] Couldn't create or write in logfile %s. Will skip logging in the future" % self.filename)
            return (-1)
"""
This file contains code to index and cache the file lookups for various datasets




"""


from pathlib import Path
from abc import ABC
from typing import Optional
import os, json, threading, tempfile, fcntl

class Cachable(ABC):
  
  CACHE_PATH = Path(__file__).parent.parent

  def __init__(self,dataset_name: Optional[str] = None):
    super().__init__()
    
    self.root_path = Path(Cachable.CACHE_PATH)
    self.cache_path = self.root_path / ".dataset_cache"


    os.makedirs(self.cache_path, exist_ok=True)

    if dataset_name:
      self.key = f"{dataset_name}"
    else:
      self.key = self.__class__.__name__.lower()

    self._lock = threading.Lock()
    self.cached = self._get()
      

  def _get(self) -> dict:
    file = self.cache_path / (self.key + ".json") 
    
    if not file.exists():
      return dict()

    with open(file, "r") as fp:
      file = json.load(fp=fp)

    return file  


  def save(self):
    file = self.cache_path / (self.key + ".json") 
    lock_file = self.cache_path / (self.key + ".lock")
    with self._lock:
      # cross-process lock
      with open(lock_file, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_path, prefix=self.key + "_", suffix=".tmp")
        try:
          with os.fdopen(fd, "w") as fp:
            json.dump(self.cached, fp=fp)
            fp.flush()
            os.fsync(fp.fileno())
          os.replace(tmp_path, file)
        finally:
          # release lock
          fcntl.flock(lf, fcntl.LOCK_UN)
  
  def cache(self, key : str, func) -> dict:
    with self._lock:
      if (key not in self.cached):
        self.cached[key] = func()
      return self.cached[key]
    





# Load Dependencies

import numpy as np
from abc import ABC, abstractmethod
import pandas as pd

class BaseRecommender(ABC):
  @abstractmethod
  def fit(self, train_df:pd.DataFrame, movies_df:pd.DataFrame, **kwargs):
    pass

  @abstractmethod
  def predict(self, user_id:np.ndarray, movie_id:np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  def recommend(self, user_id:int, n:int=10) -> pd.DataFrame:
    pass
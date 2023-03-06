from models import Models
from data_loader import DataLoader

class Experiment: 
  def __init__(self, path_to_data, model_name) -> None:
    self.model = Models(model_name=model_name)
    self.data_loader = DataLoader(path=path_to_data)
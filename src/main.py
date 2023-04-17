import torch
from constants import constants

from data_loader import DataLoader
from models import Models
from utils import Utils
from sklearn.cluster import KMeans
from experiment import Experiment
from figures import Figures
from metric import Metric
import pickle
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np

def test_run() :
  DL = DataLoader(constants.absa_file_path)
  interval = slice(0, 32)
  text_data, true_labels = DL.get_data_with_labels()

  ML = Models(constants.KB_bert, constants.KB_bert)
  embeds, encoded_input = ML.process(text_data[interval])
  embeds = normalize(embeds.last_hidden_state[:, 0, :].detach().numpy())

  nr_clusters = 5
  kmeans = KMeans(init='k-means++', n_clusters=nr_clusters)
  kmeans.fit(embeds)
  print('Cluster labels: ', kmeans.labels_)
  print('Real labels: ', true_labels[interval])

  # print(np.sum(embeds, axis=1))

  metric = Metric(embeds, true_labels=true_labels[interval], cluster_labels=kmeans.labels_)
  # print(metric.compute_all())
  cm = metric.compute_contingency_matrix()
  # f = pd.DataFrame(cm)
  Figures.contingency_matrix_figure(cm)
  # print(f)
  


def train():
  models = [constants.KBLab_bert, constants.KBLab_albert, constants.KBLab_sbert, constants.AF_bert, constants.ML_bert, constants.ML_sbert]
  DL = DataLoader(constants.absa_file_path)
  ML = Models(constants.KBLab_bert, constants.KBLab_bert)
  data = Utils.from_series_to_list(DL.get_data_by_col_name('text'))
  epochs = [4, 4, 4, 4, 4, 4]
  lr = [5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5]
  batch_size = [32, 16, 32, 32, 16, 16]
  is_save = True
  hyperparam_idx = 0
  for model in models:
    ML = Models(model, model)
    ML.fine_tune_MLM(data, epochs=epochs[hyperparam_idx], lr=lr[hyperparam_idx], batch_size=batch_size[hyperparam_idx], is_save=is_save)
    hyperparam_idx += 1
      
def get_max_seq_len():
  DL = DataLoader(constants.absa_file_path)
  data = Utils.from_series_to_list(DL.get_data_by_col_name('text'))
  data = [d.split(' ') for d in data]
  print(max(list(map(len, data))))

def get_distribution_figures(file_name):
  with open(file_name, 'rb') as f:
      distributions = pickle.load(f)
  for model_name, distribution in distributions.items():
    Figures.true_labels_per_cluster_figure(distribution, model_name=model_name)

def main():
  # train()
  # test_run()
  # Figures.search_hyperparam_figure()
  # Experiment.run_experiment_base(is_interval=False)
  # get_distribution_figures('label_distribution_base.pkl')
  # Figures.evaluation_metric_figure()
  print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
  print(f"CUDA version: {torch.version.cuda}")
    
  # Storing ID of current CUDA device
  cuda_id = torch.cuda.current_device()
  print(f"ID of current CUDA device: {torch.cuda.current_device()}")
          
  print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
  pass


if __name__ == '__main__': 
  main()

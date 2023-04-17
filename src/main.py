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
  intervals = [slice(0, 64), slice(64, 128), slice(128, 256)]
  #intervals = [slice(0, 64)]
  #interval = slice(0, 2048)
  text_data, true_labels = DL.get_data_with_labels()
  embed_data = []
  ML = Models(constants.KB_bert, constants.KB_bert)
  for interval in intervals:
    embeds, encoded_input = ML.process(text_data[interval])
    embeds = torch.Tensor.cpu(embeds.last_hidden_state[:, 0, :])
    embeds = embeds.detach().numpy()
    embed_data.append(embeds)
    torch.cuda.empty_cache()
  #embeds = embeds.last_hidden_state[:, 0, :].detach().numpy()
  embed_data = np.concatenate(embed_data)
  nr_clusters = 5
  kmeans = KMeans(init='k-means++', n_clusters=nr_clusters)
  kmeans.fit(embed_data)
  print('Cluster labels: ', kmeans.labels_)
  print('Real labels: ', true_labels[slice(0, 128)])

  # print(np.sum(embeds, axis=1))

  metric = Metric(embeds, true_labels=true_labels[slice(0, 128)], cluster_labels=kmeans.labels_)
  # print(metric.compute_all())
  cm = metric.compute_contingency_matrix()
  f = pd.DataFrame(cm)
  #Figures.contingency_matrix_figure(cm)
  print(f)
  


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
  test_run()
  # Figures.search_hyperparam_figure()
  # Experiment.run_experiment_base(is_interval=False)
  # get_distribution_figures('label_distribution_base.pkl')
  # Figures.evaluation_metric_figure()
  pass


if __name__ == '__main__': 
  main()

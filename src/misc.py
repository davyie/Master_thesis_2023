import torch
from constants import constants

import random as r
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
from clustering import Clustering
from kmeans_experiment import kMeans_Experiment
from overlim import OverLim
from collections import Counter
import matplotlib.pyplot as plt
def print_results():
  base_absa = Utils.read_file('result_absa/base/eval_metrics_base.pkl')
  improv_absa = Utils.read_file('result_absa/improv/eval_metrics_improv.pkl')

  base_sst = Utils.read_file('result_sst/base/eval_metrics_base.pkl')
  improv_sst = Utils.read_file('result_sst/improv/eval_metrics_improv.pkl')

  base_absa_metrics = {}
  improv_absa_metrics = {}
  diff_absa = {}
  diff_procent_absa = {}

  base_sst_metrics = {}
  improv_sst_metrics = {}
  diff_sst = {}
  diff_procent_sst = {}
  significant_digits = 3

  for model in constants.model_names:
    m1, m2, diff, diff_procent = [], [], [], []
    for v1, v2 in zip(list(base_absa[model].values()), list(improv_absa[model].values())):
      r1 = round(v1, significant_digits)
      r2 = round(v2, significant_digits)
      m1.append(r1)  
      m2.append(r2)
      diff.append(round(r2 - r1, significant_digits))
      diff_procent.append(round((r2 - r1) / r1, significant_digits))
    base_absa_metrics[model] = m1
    improv_absa_metrics[model] = m2
    diff_absa[model] = diff
    diff_procent_absa[model] = diff_procent

  for model in constants.model_names:
    m1, m2, diff, diff_procent = [], [], [], []
    for v1, v2 in zip(list(base_sst[model].values()), list(improv_sst[model].values())):
      r1 = round(v1, significant_digits)
      r2 = round(v2, significant_digits)
      m1.append(r1)
      m2.append(r2)
      diff.append(round(r2 - r1, significant_digits))
      diff_procent.append(round((r2 - r1) / r1, significant_digits))
    base_sst_metrics[model] = m1
    improv_sst_metrics[model] = m2
    diff_sst[model] = diff
    diff_procent_sst[model] = diff_procent

  
  keys = base_absa['KB BERT'].keys()

  print('Absa')
  print(keys)
  for k, v in base_absa_metrics.items():
    print(k, v)
  print('-----------------------------------')
  for k, v in improv_absa_metrics.items():
    print(k, v)  
  print('-----------------------------------')
  print('Absolute diff')
  for k, v in diff_absa.items():
    print(k, v)
  print('-----------------------------------')
  print('Procentuell diff')
  for k, v in diff_procent_absa.items():
    print(k, v)

  print('-----------------------------------')
  print('SST')
  print(keys)
  for k, v in base_sst_metrics.items():
    print(k, v)
  print('-----------------------------------')
  for k, v in improv_sst_metrics.items():
    print(k, v)
  print('-----------------------------------')
  print('Absolute diff')
  for k, v in diff_sst.items():
    print(k, v)
  print('-----------------------------------')
  print('Procentuell diff')
  for k, v in diff_procent_sst.items():
    print(k, v)

def mean_pooling(model_output, attention_mask):
      token_embeddings = model_output[0] #First element of model_output contains all token embeddings
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def test_run(title, folder, text_data, labels, model_name, tokenizer) :
  
  interval = slice(0, 32) 
  intervals = []
  batch_size = 64
  current = 0
  while True:
    if current + batch_size > len(text_data):
      intervals.append(slice(current, len(text_data)))
      break
    intervals.append(slice(current, current + batch_size))
    current += batch_size
    # if len(intervals) > 2:
    #   break

  embed_data = []
  ML = Models(model_name, tokenizer)

  if model_name == constants.ML_sbert_finetuned or model_name == constants.KB_sbert_finetuned:
    for interval in intervals:
      print(interval)
      embeds, encoded_input = ML.process(text_data[interval])
      embeds = mean_pooling(embeds, encoded_input['attention_mask']).detach().numpy()
      embed_data.append(embeds)
    list_of_labels = list(labels)
    initial_points, encoded_inputs = ML.process(list_of_labels)
    initial_points = mean_pooling(initial_points, encoded_inputs['attention_mask'])
  else: 
    for interval in intervals: 
      print(interval)
      embeds, _ = ML.process(text_data[interval])
      embed_data.append(embeds.last_hidden_state[:, 0, :].detach().numpy())
    list_of_labels = list(labels)
    initial_points, _ = ML.process(list_of_labels)
    initial_points = initial_points.last_hidden_state[:, 1, :].detach().numpy()

  embed_data = np.concatenate(embed_data)

  file = open(folder + title + '_embeddings.pkl', 'wb')
  pickle.dump(embed_data, file)
  file.close()

  file = open(folder + title + '_initial_points_embeddings.pkl', 'wb')
  pickle.dump(initial_points, file)
  file.close()
  

def train(data, folder):
  models = [constants.KB_bert, constants.KB_albert, constants.KB_sbert, constants.AF_bert, constants.ML_bert, constants.ML_sbert]
  epochs = [4, 4, 4, 4, 4, 4]
  lr = [5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5]
  batch_size = [32, 16, 32, 32, 16, 16]
  is_save = True
  hyperparam_idx = 0
  for model in models:
    ML = Models(model, model)
    ML.fine_tune_MLM(data, epochs=epochs[hyperparam_idx], lr=lr[hyperparam_idx], batch_size=batch_size[hyperparam_idx], folder=folder, is_save=is_save)
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
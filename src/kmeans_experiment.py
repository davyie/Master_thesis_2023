import pickle 
from figures import Figures
from utils import Utils
from constants import constants
from sklearn.cluster import KMeans
from metric import Metric
import torch
import numpy as np 

class kMeans_Experiment:
  def __init__(self) -> None:
    print('Init k-Means Experiment...')
    pass

  def run_base_kMeans(self, dataset_labels, dataset_labels_ticks, cluster_ids, embeddings_name, embeddings_folder, number_of_clusters, true_labels_filename, result_folder):
    true_labels = Utils.read_file(embeddings_folder + true_labels_filename)
    result_metrics = {}
    for (i, embedding_file_name) in enumerate(embeddings_name):
      embedding = Utils.read_file(embeddings_folder + embedding_file_name)
      kmeans = KMeans(init='k-means++', n_clusters=number_of_clusters)
      kmeans.fit(embedding)

      metric = Metric(embedding, true_labels=true_labels, cluster_labels=kmeans.labels_)
      metrics = metric.compute_all()
      cm = metric.compute_contingency_matrix()

      # Save Contingency Matrix 
      Figures.contingency_matrix_figure(cm, constants.model_names[i], result_folder, dataset_labels, dataset_labels_ticks, cluster_ids)

      result_metrics[constants.model_names[i]] = metrics
      print(embedding_file_name)
      print(constants.model_names[i])
      print('------------------------------')
    # Save the metrics
    file = open(result_folder + 'eval_metrics_base.pkl', 'wb')
    pickle.dump(result_metrics, file)
    file.close()

    # Get evaluation figures 
    Figures.evaluation_metric_figure(result_folder + 'eval_metrics_base.pkl', result_folder) 
    

  def run_improv_kMeans(self, dataset_labels, dataset_labels_ticks, cluster_ids, embeddings_name, embeddings_folder, number_of_clusters, true_labels_filename, result_folder):
    true_labels = Utils.read_file(embeddings_folder + true_labels_filename)

    result_metrics = {}
    for (i, embedding_file_name) in enumerate(embeddings_name):
      model_name = constants.model_names[i]
      initial_points_file_name = constants.initial_points_embeddings[i]

      embedding = Utils.read_file(embeddings_folder + embedding_file_name)
      initial_points_embeddings = Utils.read_file(embeddings_folder + initial_points_file_name)
      initial_points_embeddings = initial_points_embeddings if not torch.is_tensor(initial_points_embeddings) else initial_points_embeddings.detach().numpy()
      print(initial_points_embeddings.shape)

      kmeans = KMeans(init=initial_points_embeddings, n_clusters=number_of_clusters)
      kmeans.fit(embedding)

      metric = Metric(embedding, true_labels=true_labels, cluster_labels=kmeans.labels_)
      metrics = metric.compute_all()
      cm = metric.compute_contingency_matrix()

      # Save Contingency Matrix 
      Figures.contingency_matrix_figure(cm, constants.model_names[i], result_folder, dataset_labels, dataset_labels_ticks, cluster_ids)
      # Save Metrics
      result_metrics[model_name] = metrics
      print(embedding_file_name)
      print(model_name)
      # print(metrics)
      print('------------------------------')
    
    file = open(result_folder + 'eval_metrics_improv.pkl', 'wb')
    pickle.dump(result_metrics, file)
    file.close()

    # Get evaluation figures 
    Figures.evaluation_metric_figure(result_folder + 'eval_metrics_improv.pkl', result_folder) 
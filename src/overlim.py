import pickle

from sklearn.cluster import KMeans
import torch
from data_loader import DataLoader
from constants import constants
from figures import Figures
from metric import Metric
from models import Models
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from utils import Utils 

class OverLim():
    def __init__(self) -> None:
        pass
    
    def get_embeddings(title, folder, model, tokenizer, labels):
        DL = DataLoader(constants.absa_file_path)
        train_data, train_labels = DL.read_json(constants.overlim_train_file_path)
        train_data, train_labels = train_data[0:2300], train_labels[0:2300]
        val_data, val_labels = DL.read_json(constants.overlim_val_file_path)
        test_data, test_labels = DL.read_json(constants.overlim_test_file_path)

        data, labels = train_data + val_data + test_data, train_labels + val_labels + test_labels
        title = 'ML SBERT'
        folder = 'finetuned_embeddings_sst/'
        model = constants.ML_sbert_finetuned_sst
        tokenizer = constants.ML_sbert
        labels = constants.overlime_sst_labels.values()

        # test_run(title=title, text_data=data, labels=labels, model_name=model, tokenizer=tokenizer, folder=folder)

    def run_kmeans_base(self, embeddings_name):
      true_labels = Utils.read_file('finetuned_embeddings_sst/true_labels.pkl')
      result_folder = 'result-sst/'

      result_metrics = {}
      for (i, embedding_file_name) in enumerate(embeddings_name):
        embedding = Utils.read_file(constants.embedding_folder_sst + embedding_file_name)
        nr_clusters = 2
        kmeans = KMeans(init='k-means++', n_clusters=nr_clusters)
        kmeans.fit(embedding)

        metric = Metric(embedding, true_labels=true_labels, cluster_labels=kmeans.labels_)
        metrics = metric.compute_all()
        cm = metric.compute_contingency_matrix()
        # true_label_distribution = Utils.get_true_labels_per_cluster(true_labels, kmeans.labels_)
        
        # Save Contingency Matrix 
        Figures.contingency_matrix_figure(cm, constants.model_names[i], result_folder)

        result_metrics[constants.model_names[i]] = metrics
        print(embedding_file_name)
        print(constants.model_names[i])
        # print(metrics)
        print('------------------------------')
      # Save the metrics
      print(result_metrics)
      file = open(result_folder + 'eval_metrics_base.pkl', 'wb')
      pickle.dump(result_metrics, file)
      file.close()

      # Get evaluation figures 
      Figures.evaluation_metric_figure(result_folder + 'eval_metrics_base.pkl', result_folder) 

    def run_improv_kMeans(self, embeddings_name, initial_embeddings_name):
      true_labels = Utils.read_file('finetuned_embeddings_sst/true_labels.pkl')
      result_folder = 'result-sst/'

      result_metrics = {}
      for (i, embedding_file_name) in enumerate(embeddings_name):
        model_name = constants.model_names[i]
        initial_points_file_name = constants.initial_points_embeddings_sst[i]


        embedding = Utils.read_file(constants.embedding_folder_sst + embedding_file_name)
        initial_points_embeddings = Utils.read_file(constants.embedding_folder_sst + initial_points_file_name)
        initial_points_embeddings = initial_points_embeddings if not torch.is_tensor(initial_points_embeddings) else initial_points_embeddings.detach().numpy()
        print(initial_points_embeddings.shape)

        nr_clusters = 2
        kmeans = KMeans(init=initial_points_embeddings, n_clusters=nr_clusters)
        kmeans.fit(embedding)

        metric = Metric(embedding, true_labels=true_labels, cluster_labels=kmeans.labels_)
        metrics = metric.compute_all()
        cm = metric.compute_contingency_matrix()

        # Save Contingency Matrix 
        Figures.contingency_matrix_figure(cm, model_name, result_folder)
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

    def print_samples(self):
      # sample_indicies = [r.randint(0, len(data)) for _ in range(0, 5)]
      
      # print('Max length: {}'.format(max([len(d) for d in data])))
      # print('Sample indices: ', sample_indicies)

      # print('\n')
      # print('***********************************************************************************************')
      # print('Label    |  Text')
      # print('-----------------------------------------------------------------------------------------------')
      # for i in sample_indicies:
      #   print(constants.overlime_sst_labels[labels[i]].ljust(label_max_len), '| ' , data[i][0:75])
      # print('***********************************************************************************************')
      # print('\n')

      labels = ['negative', 'positive', 'positive', 'negative', 'positive']
      samples = ['waiting for the video', 'will probably enjoy themselves', 'hits the bullseye', 'Further sad evidence that Tom Tykwer , head of resonant and sense', 'whimsical and relevant today']
      # label_max_len = max(map(len, [constants.overlime_sst_labels[i] for i in labels]))  
      print('\n')
      print('***********************************************************************************************')
      print('Label    |  Text')
      print('-----------------------------------------------------------------------------------------------')
      for (label, text) in zip(labels, samples):
        print(label, '| ' , text[0:75])
      print('***********************************************************************************************')
      print('\n')
    def print_data_distribution(self):
        DL = DataLoader(constants.absa_file_path)
        train_data, train_labels = DL.read_json(constants.overlim_train_file_path)
        val_data, val_labels = DL.read_json(constants.overlim_val_file_path)
        test_data, test_labels = DL.read_json(constants.overlim_test_file_path)

        data, labels = train_data[0:2300] + val_data + test_data, train_labels[0:2300] + val_labels + test_labels

        constants.overlime_sst_labels
        counts = Counter(labels)
        print(counts)
        plt.bar([0, 1], counts.values(), align='center', alpha=0.5, color=['#F96140', '#7EF940'])
        plt.xticks([1, 0], ['positive', 'negative'])
        plt.ylabel('Count')
        plt.xlabel('Opinion')
        plt.title("OverLim SST (subset) - Data distribution")
        plt.show()
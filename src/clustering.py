from sklearn.cluster import KMeans
import torch
from constants import constants

from data_loader import DataLoader
from models import Models
from utils import Utils 

class Clustering: 
  def __init__(self, n_clusters, init_points=[]) -> None:
      # use init_points later 
      self.kmeans = KMeans(init='k-means++', n_clusters=n_clusters, algorithm='lloyd')
  
  def get_labels(self):
      '''
      This method returns the labels for each datapoint
      '''
      return self.kmeans.labels_
  
  def cluster_bert(model_name, path_to_data, interval = slice(0, 16), is_initial_points = False):
    '''
    This clustering method works for 
    - KBLab BERT 
    - AF BERT 
    - ML BERT 
    '''
    DL = DataLoader(path_to_data)
    ML = Models(model_name=model_name)
    text_data, true_labels = DL.get_data_with_labels()
    # data = Utils.from_series_to_list(text_data)[interval] # Get subset of texts
    model_output, _ = ML.process(text_data[interval])
    embeds = model_output.last_hidden_state[:, 0, :] # Get embedding. Grab first vector
    
    list_of_labels = list(constants.absa_labels.values())
    initial_points, _ = ML.process(list_of_labels)
    initial_points = initial_points.last_hidden_state[:, 0, :].detach().numpy()
    

    nr_clusters = 5
    kmeans = KMeans(init=initial_points if is_initial_points else 'k-means++', n_clusters=nr_clusters)
    kmeans.fit(embeds.detach().numpy())

    return embeds.detach().numpy(), true_labels[interval], kmeans.labels_

  def cluster_sbert(model_name, path_to_data, interval = slice(0, 16)):
    '''
    This works for Sentence BERT 
    - KBLab SBERT 
    - ML SBERT 
    '''
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
      token_embeddings = model_output[0] #First element of model_output contains all token embeddings
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    DL = DataLoader(path_to_data)
    ML = Models(model_name=model_name)
    data = Utils.from_series_to_list(DL.get_data_by_col_name('text'))[0:64] # Get subset of texts
    model_output, encoded_inputs = ML.process(data)
    embeds = mean_pooling(model_output, encoded_inputs['attention_mask'])

    nr_clusters = 5
    kmeans = KMeans(init='k-means++', n_clusters=nr_clusters)
    kmeans.fit(embeds.detach().numpy())
    print(kmeans.labels_)

  def cluster_albert(model_name, path_to_data, interval = slice(0, 16)):
    DL = DataLoader(path=path_to_data)
    ML = Models(model_name=model_name)
    data = Utils.from_series_to_list(DL.get_data_by_col_name('text'))[0:64]
    model_output, encoded_inputs = ML.process(data)
    embeds = model_output.last_hidden_state[:, 0, :]

    nr_clusters = 5
    kmeans = KMeans(init='k-means++', n_clusters=nr_clusters)
    kmeans.fit(embeds.detach().numpy())
    print(kmeans.labels_)
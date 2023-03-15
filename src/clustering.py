from sklearn.cluster import KMeans
import torch
from constants import constants

from data_loader import DataLoader
from models import Models
from utils import Utils 

class Clustering: 
  '''
  Use this class to run clustering. 
  '''
  def __init__(self, n_clusters, init_points=[]) -> None:
      # use init_points later 
      self.kmeans = KMeans(init='k-means++', n_clusters=n_clusters, algorithm='lloyd')
  
  def get_labels(self):
      '''
      This method returns the labels for each datapoint
      '''
      return self.kmeans.labels_
  
  def cluster_bert(model_name, tokenizer_name, path_to_data, is_interval, is_improve=False):
    '''
    This clustering method works for 
    - KBLab BERT 
    - AF BERT 
    - ML BERT 
    '''
    DL = DataLoader(path_to_data)
    ML = Models(model_name=model_name, tokenizer_name=tokenizer_name)
    text_data, true_labels = DL.get_data_with_labels()
    # data = Utils.from_series_to_list(text_data)[interval] # Get subset of texts
    if is_interval: 
      model_output, _ = ML.process(text_data[0:16])
    else:
      model_output, _ = ML.process(text_data)
    embeds = model_output.last_hidden_state[:, 0, :] # Get embedding. Grab first vector
    
    list_of_labels = list(constants.absa_labels.values())
    initial_points, _ = ML.process(list_of_labels)
    initial_points = initial_points.last_hidden_state[:, 0, :].detach().numpy()

    nr_clusters = 5
    kmeans = KMeans(init=initial_points if is_improve else 'k-means++', n_clusters=nr_clusters)
    kmeans.fit(embeds.detach().numpy())

    return embeds.detach().numpy(), true_labels[0:16] if is_interval else true_labels, kmeans.labels_

  def cluster_sbert(model_name, tokenizer, path_to_data, is_interval=False, is_improve=False):
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
    ML = Models(model_name=model_name, tokenizer_name=tokenizer)
    data, true_labels = DL.get_data_with_labels()
    if is_interval: 
      model_output, encoded_inputs = ML.process(data[0:16])
    else:
      model_output, encoded_inputs = ML.process(data)
    # print(model_output, encoded_inputs)
    embeds = mean_pooling(model_output, encoded_inputs['attention_mask'])

    list_of_labels = list(constants.absa_labels.values())
    initial_points, encoded_inputs = ML.process(list_of_labels)
    initial_points = mean_pooling(initial_points, encoded_inputs['attention_mask'])

    nr_clusters = 5
    kmeans = KMeans(init=initial_points.detach().numpy() if is_improve else 'k-means++', n_clusters=nr_clusters)
    kmeans.fit(embeds.detach().numpy())

    return embeds.detach().numpy(), true_labels[0:16] if is_interval else true_labels, kmeans.labels_

  def cluster_albert(model_name, tokenizer, path_to_data, is_interval = False, is_improve=False):
    DL = DataLoader(path=path_to_data)
    ML = Models(model_name=model_name, tokenizer_name=tokenizer)
    data, true_labels = DL.get_data_with_labels()
    if is_interval: 
      model_output, _ = ML.process(data[0:16])
    else:
      model_output, _ = ML.process(data)
    embeds = model_output.last_hidden_state[:, 0, :]
    
    list_of_labels = list(constants.absa_labels.values())
    initial_points, _ = ML.process(list_of_labels)
    initial_points = initial_points.last_hidden_state[:, 0, :].detach().numpy()

    nr_clusters = 5
    kmeans = KMeans(init=initial_points if is_improve else 'k-means++', n_clusters=nr_clusters)
    kmeans.fit(embeds.detach().numpy())
    
    return embeds.detach().numpy(), true_labels[0:16] if is_interval else true_labels, kmeans.labels_

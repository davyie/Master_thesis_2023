from collections import Counter
from itertools import combinations
import numpy as np 
class Utils:
    
  def from_tensor_to_nparray(tensor):
      '''
      This method helps us convert a tensor to a np array which is used in 
      skleanr K-means algorithm. 
      Use this before running K-means.
      @param tensor - a tensor object 
      '''
      return tensor.detach().numpy()
  
  def get_cls_from_batch(data):
     '''
     This method extracts the CLS token embedding from a batch of processed sequences.
     The dimension of data is [batch, #token, hidden_size].
     The solution is to keep the batch and hidden size but we want the first. 
     Therefore it is a 0 at the middle. 
     Use this when obtained BERT embeddings. 
     @param data - This argument has the type tensor 
     '''
     return data.last_hidden_state[:, 0, :]
  
  def get_str_list_of_labels(label_dict):
     '''
     This method returns a str list of labels. 
     @param label_dict - this is the label dictionary from constants.
     '''
     return list(label_dict.values())
  
  def from_series_to_list(series):
     return series.tolist()

  def get_confusion_matrix(true_labels, cluster_labels):
    '''
      This method computes confusion matrix. The computation is done for 
      each PAIR in the dataset. 
      The out is as follows: 
      [ 
        [<True neg>, <False Neg>],
        [<False Pos, <True Pos>] 
      ]
      True negative is defined as when true labels DISAGREE and algorithm label DISAGREE. 
      True positive is defined as wehn true labels AGREE and algorithm label AGREE. 
      False negative is defined as when true labels AGREE and algorithm label DISAGREE.
      False positive is defined as when true labels DISAGREE and algorithm label AGREE. 
    '''
    m = len(cluster_labels)
    n = len(true_labels)
    if n != m: # Error message 
      print("Different lengths! C_labels: {} | T_labels: {}".format(m, n))
      return 
    confusion_matrix = [[0 for _ in range(2)] for _ in range(2)]
    m = len(cluster_labels)
    n = len(true_labels)
    pairwise_points = list(combinations(range(n - 1), 2))
    for (a, b) in pairwise_points: 
      c_a_label = cluster_labels[a]
      t_a_label = true_labels[a]
      c_b_label = cluster_labels[b]
      t_b_label = true_labels[b] # cluster labels start with 1 
      if c_a_label == c_b_label and t_a_label == t_b_label: # True positive. Agree on cluster and label
        confusion_matrix[1][1] += 1
      elif c_a_label == c_b_label and t_a_label != t_b_label: # False positive 
        confusion_matrix[1][0] += 1
      elif c_a_label != c_b_label and t_a_label == t_b_label: # False negative. 
        confusion_matrix[0][1] += 1
      else: # True negative 
        confusion_matrix[0][0] += 1 
    return confusion_matrix 
        
    
  def print_to_file(filename, metrics):
    with open(filename, 'w') as out:
      for (name, metric) in metrics.items():
        out.write( "{}: {}\n".format(name, metric))
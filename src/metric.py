from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, rand_score, homogeneity_completeness_v_measure
from utils import Utils

class Metric:
  def __init__(self, data, true_labels, cluster_labels) -> None:
    self.confusion_matrix = Utils.get_confusion_matrix(true_labels=true_labels, cluster_labels=cluster_labels)

    self.TP = self.confusion_matrix[1][1]
    self.FP = self.confusion_matrix[1][0]
    self.TN = self.confusion_matrix[0][0]
    self.FN = self.confusion_matrix[0][1]

    self.true_labels = true_labels
    self.cluster_labels = cluster_labels
    self.data = data
    
  def compute_accuracy(self):
    '''
      This is the same as Rand index in terms of clustering. 
    '''
    return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
  
  def compute_silhouette(self):
    return silhouette_score(self.data, self.cluster_labels)
  
  def compute_calinski_harabasz_score(self):
    return calinski_harabasz_score(self.data, self.cluster_labels)
  
  def compute_davies_bouldin_score(self):
    return davies_bouldin_score(self.data, self.cluster_labels)
  
  def compute_rand_index(self):
    return rand_score(self.true_labels, self.cluster_labels)
  
  def compute_v_measure(self):
    self.h, self.c, self.v = homogeneity_completeness_v_measure(self.true_labels, self.cluster_labels)
    return self.h, self.c, self.v
  
  def compute_all(self):
    self.compute_v_measure()
    return {"Rand": self.compute_accuracy(), "Silhouette": self.compute_silhouette(), "CHS": self.compute_calinski_harabasz_score(), "DBS": self.compute_davies_bouldin_score(), 'Homogeneity': self.h, 'Completeness': self.c, 'V-measure': self.v }
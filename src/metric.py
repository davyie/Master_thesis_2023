from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
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
  
  def compute_precision(self):
    '''
      The number of correctly predicted datapoints given all positive. 
    '''
    return self.TP / (self.TP + self.FP)
  
  def compute_recall(self):
    '''
      This computes the number of correctly predicted positives out of 
      actual positives. 
    '''
    return self.TP / (self.TP + self.FN)

  def compute_F1(self):
    recall_score = self.compute_recall()
    precision_score = self.compute_precision()
    return 2 * precision_score * recall_score / (precision_score + recall_score)
  
  def compute_silhouette(self):
    return silhouette_score(self.data, self.cluster_labels)
  
  def compute_calinski_harabasz_score(self):
    return calinski_harabasz_score(self.data, self.cluster_labels)
  
  def compute_davies_bouldin_score(self):
    return davies_bouldin_score(self.data, self.cluster_labels)
  
  def compute_all(self):
    return {"Accuracy": self.compute_accuracy(), "Precision": self.compute_precision(), "Recall": self.compute_recall(), "F1": self.compute_F1(), "Silhouette": self.compute_silhouette(), "CHS": self.compute_calinski_harabasz_score(), "DBS": self.compute_davies_bouldin_score()}
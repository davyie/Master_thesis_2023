from clustering import Clustering
from constants import constants
from metric import Metric
from utils import Utils

class Experiment: 
  def KB_BERT_Experiment_BASE(model_name, tokenizer_name, path_to_data):
    data, true_labels, cluster_labels = Clustering.cluster_bert(model_name, tokenizer_name, path_to_data)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    # print('Without improvement: ', M.compute_accuracy(), M.compute_precision(), M.compute_recall(), M.compute_silhouette(), M.compute_calinski_harabasz_score())
    metrics = M.compute_all()
    Utils.print_to_file("result/KB_BERT_BASE.txt", metrics)
  
  def KB_BERT_Experiment_IMPROV():
    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.KBLab_bert, constants.absa_file_path, is_initial_points=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics = M.compute_all()
    Utils.print_to_file("result/KB_BERT_IMPROV.txt", metrics)

  def AF_BERT_Experiment_BASE():
    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.AF_bert, constants.absa_file_path)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    # print('Without improvement: ', M.compute_accuracy(), M.compute_precision(), M.compute_recall(), M.compute_silhouette(), M.compute_calinski_harabasz_score())
    metrics = M.compute_all()
    Utils.print_to_file("result/AF_BERT_BASE.txt", metrics)

  def AF_BERT_Experiment_IMPROV():
    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.AF_bert, constants.absa_file_path, is_initial_points=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics = M.compute_all()
    Utils.print_to_file("result/AF_BERT_IMPROV.txt", metrics)

  def ML_BERT_Experiment_BASE():
    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.ML_bert, constants.absa_file_path)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    # print('Without improvement: ', M.compute_accuracy(), M.compute_precision(), M.compute_recall(), M.compute_silhouette(), M.compute_calinski_harabasz_score())
    metrics = M.compute_all()
    Utils.print_to_file("result/ML_BERT_BASE.txt", metrics)

  def ML_BERT_Experiment_IMPROV():
    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.ML_bert, constants.absa_file_path, is_initial_points=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics = M.compute_all()
    Utils.print_to_file("result/ML_BERT_IMPROV.txt", metrics)
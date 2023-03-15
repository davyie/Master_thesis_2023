from clustering import Clustering
from constants import constants
from metric import Metric
from utils import Utils
import pickle

class Experiment: 
  def run_experiment_base():
    is_interval = True
    metrics = {}

    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.KB_bert_finetuned, constants.KB_bert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB bert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_albert(constants.KB_albert_finetuned, constants.KB_albert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB albert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_sbert(constants.KB_sbert_finetuned, constants.KB_sbert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB sbert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.AF_bert_finetuned, constants.AF_bert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['AF bert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.ML_bert_finetuned, constants.ML_bert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['ML bert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_sbert(constants.ML_sbert_finetuned, constants.ML_sbert, constants.absa_file_path, is_interval=is_interval)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['ML sbert'] = M.compute_all()
    
    print(metrics)
    file = open('eval_metrics_base.pkl', 'wb')
    pickle.dump(metrics, file)
    file.close()

  def run_experiment_improve():
    is_interval = False
    metrics = {}
    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.KB_bert_finetuned, constants.KB_bert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB bert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_albert(constants.KB_albert_finetuned, constants.KB_albert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB albert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_sbert(constants.KB_sbert_finetuned, constants.KB_sbert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['KB sbert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.AF_bert_finetuned, constants.AF_bert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['AF bert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_bert(constants.ML_bert_finetuned, constants.ML_bert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['ML bert'] = M.compute_all()

    data, true_labels, cluster_labels = Clustering.cluster_sbert(constants.ML_sbert_finetuned, constants.ML_sbert, constants.absa_file_path, is_interval=is_interval, is_improve=True)
    M = Metric(data, true_labels=true_labels, cluster_labels=cluster_labels)
    metrics['ML sbert'] = M.compute_all()
    
    print(metrics)
    file = open('eval_metrics_improv.pkl', 'wb')
    pickle.dump(metrics, file)
    file.close()
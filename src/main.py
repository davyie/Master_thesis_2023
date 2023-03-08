import torch
from constants import constants

from data_loader import DataLoader
from models import Models
from utils import Utils
from transformers import AutoModel,AutoTokenizer, AlbertModel, AlbertTokenizer
from sentence_transformers import SentenceTransformer
from clustering import Clustering
from sklearn.cluster import KMeans
from experiment import Experiment

# texts = ['En mening', 'En annan mening', 'En ännu längre mening']

def test_run() :
  DL = DataLoader(constants.absa_file_path)
  ML = Models(constants.AF_bert)
  # data = Utils.from_series_to_list(DL.get_data_by_col_name('text'))[0:64]
  interval = slice(0, 16)
  text_data, true_labels = DL.get_data_with_labels()
  embeds, encoded_input = ML.process(text_data[interval])
  embeds = embeds.last_hidden_state[:, 0, :]

  nr_clusters = 5
  kmeans = KMeans(init='k-means++', n_clusters=nr_clusters)
  kmeans.fit(embeds.detach().numpy())
  print(kmeans.labels_)
  print(true_labels[interval])

  Utils.get_confusion_matrix(true_labels=true_labels[interval], cluster_labels=kmeans.labels_)
  # print(label_assignment)

def test_train():
  DL = DataLoader(constants.absa_file_path)
  ML = Models(constants.ML_sbert, constants.ML_sbert)
  data = Utils.from_series_to_list(DL.get_data_by_col_name('text'))[0:16]
  epochs = 3

  ML.fine_tune_MLM(data, epochs=epochs, is_save=False)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
  token_embeddings = model_output[0] #First element of model_output contains all token embeddings
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main():
  # meningar = ['Hej jag är david', 'Mitt namn är david']
  # tokenizer = AutoTokenizer.from_pretrained(constants.KBLab_sbert)
  # model = AutoModel.from_pretrained(constants.KBLab_sbert)

  # encoded_input = tokenizer(meningar, padding=True, truncation=True, return_tensors="pt")

  # with torch.no_grad():
  #   model_output = model(**encoded_input)

  # sembed = mean_pooling(model_output, encoded_input['attention_mask'])

  # print(sembed.size())
  # test_train()
  # print(dl.get_label_distribution())
  DL = DataLoader(constants.absa_file_path)
  DL.save_to_file()
  pass


if __name__ == '__main__': 
  main()

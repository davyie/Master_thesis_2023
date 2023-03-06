from enums import ModelType
from transformers import AutoModel, AutoTokenizer, AlbertTokenizer, AlbertModel

class constants:
  '''
    This class contains constants for the project 
  '''
  absa_labels = {1: 'mycket negativt', 2: 'negativt', 3: 'neutral', 4: 'positivt', 5: 'mycket positivt' }
  ski_labels = {1: 'image', 2: 'product quality', 3: 'service quality', 4: 'expectation'}

  absa_file_path = '../dataset/absabank_imm/P_annotation.tsv'

  # Model names 
  KBLab_bert = 'KBLab/bert-base-swedish-cased'
  KBLab_sbert = 'KBLab/sentence-bert-swedish-cased'
  KBLab_albert = 'KBLab/albert-base-swedish-cased-alpha'
  AF_bert = 'af-ai-center/bert-base-swedish-uncased'
  ML_bert = 'bert-base-multilingual-cased'
  ML_sbert = 'sentence-transformers/distiluse-base-multilingual-cased-v2'

  # models = {
  #   ModelType.KBLab_BERT: [AutoTokenizer.from_pretrained(KBLab_bert), AutoModel.from_pretrained(KBLab_bert)], 
  #   ModelType.AF_BERT: [AutoTokenizer.from_pretrained(AF_bert), AutoModel.from_pretrained(AF_bert)],
  #   ModelType.KBLAB_ALBERT: [AlbertTokenizer.from_pretrained()]}
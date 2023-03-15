class constants:
  '''
    This class contains constants for the project 
  '''
  absa_labels = {1: 'mycket negativt', 2: 'negativt', 3: 'neutral', 4: 'positivt', 5: 'mycket positivt' }
  ski_labels = {1: 'image', 2: 'product quality', 3: 'service quality', 4: 'expectation'}

  absa_file_path = '../dataset/absabank_imm/P_annotation.tsv'

  # Model names 
  KB_bert = 'KBLab/bert-base-swedish-cased'
  KB_sbert = 'KBLab/sentence-bert-swedish-cased'
  KB_albert = 'KBLab/albert-base-swedish-cased-alpha'
  AF_bert = 'af-ai-center/bert-base-swedish-uncased'
  ML_bert = 'bert-base-multilingual-cased'
  ML_sbert = 'sentence-transformers/distiluse-base-multilingual-cased-v2'


  KB_bert_finetuned = './models/KBLab/bert-base-swedish-cased_finetuned'
  KB_albert_finetuned = './models/KBLab/albert-base-swedish-cased-alpha_finetuned'
  KB_sbert_finetuned = './models/KBLab/sentence-bert-swedish-cased_finetuned'
  AF_bert_finetuned = './models/af-ai-center/bert-base-swedish-uncased_finetuned'
  ML_bert_finetuned = './models/bert-base-multilingual-cased_finetuned'
  ML_sbert_finetuned = './models/sentence-transformers/distiluse-base-multilingual-cased-v2_finetuned'
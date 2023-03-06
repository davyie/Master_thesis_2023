import torch 
from transformers import AutoModel, AutoTokenizer, AdamW, AutoModelForMaskedLM, AlbertTokenizer, AlbertModel, AlbertForMaskedLM
import numpy as np
from tqdm import tqdm
from constants import constants

from dataset import Dataset

class Models: 
  def __init__(self, model_name) -> None:
    '''
      This class contains different models we can use to obtain the embeddings 
    '''
    self.model_name = model_name
    if model_name == constants.KBLab_albert:
      self.tok = AlbertTokenizer.from_pretrained(model_name)
    else: 
      self.tok = AutoTokenizer.from_pretrained(model_name)
    return 
  
  def process(self, data): 
    '''
      This method will process the data and obtain embeddings 
    '''

    if self.model_name == constants.KBLab_albert:
      model = AlbertModel.from_pretrained(self.model_name)
    else:
      model = AutoModel.from_pretrained(self.model_name)
    
    inputs = self.tokenize_text(data)
    return model(**inputs), inputs
  
  def decode(self, token_id):
    return self.tok.decode(token_id)

  def tokenize_text(self, text, pad=True, truncate=True, max_len=512): 
    '''
      This method will run the transformer tokenizer such that the input fits the model. 
      The procedure is to convert each word to an id. We will return the encoded inputs. 
      Truncate and pad the text to make them the same length. 
      @param text - This is a text or list of texts 
      i.e. 'this is a sentence' or ['this is sentence', 'This is another sentence']
    '''
    return self.tok(text, padding=pad, truncation=truncate, max_length=max_len, return_tensors="pt")
    # return self.tok(text, return_tensors="pt")

  def preprocess_maskedLM_dataset(self, text_data, mask_percentage=0.15, batch_size=8):
    '''
      This method preprocesses data to MLM training scheme. 
      This method masks the data and creates the label. 
      @param text_data - a list of text. 
      @param mask_percentage - procentage of tokens being masked 
      @param batch_size - how should the data be split up for training. 
      Make sure its not greater than number of datapoints. 
    '''
    tokenized_dataset = self.tokenize_text(text_data) # Tokenize the data 
    tokenized_dataset['labels'] = tokenized_dataset['input_ids'].detach().clone() # Create the labels 
    # Create a 15% mask. Disregard CLS, SEP, PAD. mask_tensor = boolean[][]
    mask_tensor = (torch.rand(tokenized_dataset['input_ids'].shape) < mask_percentage) * (tokenized_dataset['input_ids'] != self.tok.cls_token_id) * (tokenized_dataset['input_ids'] != self.tok.sep_token_id) * (tokenized_dataset['input_ids'] != self.tok.pad_token_id) 
    tokenized_dataset['input_ids'] = torch.where(mask_tensor, self.tok.mask_token_id, tokenized_dataset['input_ids']) # replace the tokens with [MASK] where it is true

    tokenized_dataset = Dataset(tokenized_dataset)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

  def fine_tune_MLM(self, text_data, epochs, is_save=False):
    '''
      This method contains training loop for the model. 
      First it inits the model and then process the data. 
      This works for 
      Lastly, it runs the training loop and save the model if its set to save. 
      @param text_data - a list of strings 
      @param epochs - the number of epochs we train the model 
    '''
    # Load in the model. Tokenizer is already loaded in 
    if self.model_name == constants.KBLab_albert:
      model = AlbertForMaskedLM.from_pretrained(self.model_name)
    else: 
      model = AutoModel.from_pretrained(self.model_name)

    model.train() # Turn on training state

    # Prepare dataset 
    dataloader = self.preprocess_maskedLM_dataset(text_data, 0.15)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
      loop = tqdm(dataloader)
      for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

        loop.set_description('Epoch: {}'.format(epoch))
        loop.set_postfix(loss=loss.item)
    if is_save:
      model.eval()
      self.model_name = self.model_name + "_finetuned"
      model.save_pretrained("./models/" + self.model_name)
      print("./models/" + self.model_name)

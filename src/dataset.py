import torch 

class Dataset(torch.utils.data.Dataset):
      def __init__(self, encodings) -> None:
        self.encodings = encodings
      def __len__(self):
        return len(self.encodings)
      def __getitem__(self, index):
        input_ids = torch.Tensor(self.encodings['input_ids'][index])
        token_type_ids = torch.Tensor(self.encodings['token_type_ids'][index])
        attention_mask = torch.Tensor(self.encodings['attention_mask'][index])
        labels = torch.Tensor(self.encodings['labels'][index])
        return {
          'input_ids': input_ids,
          'token_type_ids': token_type_ids,
          'attention_mask': attention_mask,
          'labels': labels
        }
from web_anno_tsv import open_web_anno_tsv

def main():
  tsv = '../dataset/absabank_imm/D_annotation.tsv'

  with open_web_anno_tsv(tsv) as f: 
    for i, sentence in enumerate(f):
      print(i, sentence)
  print("Hello world")

if __name__ == '__main__': 
  main()

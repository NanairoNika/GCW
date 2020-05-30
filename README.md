# GCW
Graduation Course Work

The main data from which was data extracted: positive.csv, negative.csv, frames.json

Small utility functions are in the utility.py

Contains two methods of preprocessing:
  1. Using nltk.word_tokenize + pymorphy2
  2. Using deeppavlov
  
The classification part is nearly close to each other

For the first method use data "train_data.csv" and "test_data.csv" for the second with the _2 at the end

The best results for the first method: 85.3% by K-neighbours

The best results for the second method: 85.4% by K-neighbours

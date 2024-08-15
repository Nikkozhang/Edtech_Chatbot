import numpy as np
import nltk
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data if not already done
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
ds = load_dataset("meta-math/MetaMathQA")

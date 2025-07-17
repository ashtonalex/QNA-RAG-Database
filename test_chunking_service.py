import nltk
from nltk.tokenize import sent_tokenize

text = "This is a sentence. Here is another one!"
print(sent_tokenize(text, language='english'))
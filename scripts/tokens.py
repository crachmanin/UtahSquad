# coding: utf-8
import nltk
nltk.download('punkt')

sent = u'The cat went five to the mall. \"To the mall, it went.\" Никола Тесла'
tokens =  nltk.word_tokenize(sent)
print tokens

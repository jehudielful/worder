import nltk
from nltk.corpus import gutenberg, brown
import collections
import pandas as pd
import re
from py_translator import Translator
translator = Translator()
#nltk.download() #if u need to download smth for nltk

#Opening + tokenization
file = open('got_s01e01.srt', 'r') 
text = file.read()
file.close()
text_clean = (re.sub('[^a-zA-Z]+', ' ', text.replace('\n', ' '))).lower()

tokens = nltk.word_tokenize(text_clean)
long_tokens = []
for token in tokens:
    if len(token) > 2: #drop words less than 2 letters
        long_tokens.append(token)
ctr = collections.Counter(long_tokens) ##how often word appears in the text
ctr_df = pd.DataFrame.from_dict(dict(ctr), orient='index')
ctr_df.reset_index(inplace=True)
ctr_df.columns = ['word', 'n']

# ctr(frequency) from corpus
news_text = gutenberg.words()
ctr_df['frequency'] = 0
fdist = nltk.FreqDist(w.lower() for w in news_text)
words = list(ctr_df['word'].get_values())
for word in words:
    ctr_df.loc[ctr_df['word'] == word, 'frequency'] = fdist[word]

# names    
file = open('names.txt', 'r') 
names = file.read().split('\n')
names.extend(['Michael', 'Scott', 'Dwight', 'Schrute', 'Jim', 'Halpert', 'Pam', 'Beesly', 'Ryan', 'Howard', 
         'Andy', 'Bernard', 'Robert', 'California', 'Darryl', 'Philbin', 'Todd', 'Packer', 'Adolf', 'Hitler'])
file.close()
names = [x.lower() for x in names]

# short list of words
ctr_df_short = ctr_df[(ctr_df['frequency'] > 1) & (ctr_df['frequency'] < 10)
            & (~ctr_df['word'].isin(names))]\
    .sort_values(by='frequency')

#translation
ctr_df_short['translation'] = ' '
ctr_df_short['translation'] = ctr_df_short['word'].apply(lambda x: translator.translate(text=x,
                                                                                        dest='ru', src='en').text)
#save
ctr_df_short[['word', 'translation', 'frequency']].to_csv('words_got_s01e01.csv', encoding='utf-16', index=False,
                                                         sep='\t')
import syllapy
import pandas as pd



import math
from collections import Counter


df =  pd.read_csv('train.csv')

def word_entropy(word):
    word_len = len(word)
    freqs = Counter(word)
    entropy = -sum((freq/word_len) * math.log2(freq/word_len) for freq in freqs.values())
    return entropy

df['word_entropy'] = df['Complex_Phrase'].apply(word_entropy)


def count_syllables(word):
    return syllapy.count(word)




# Apply syllable count function to the Complex_Phrase column



df.to_csv('train.csv',index=False)

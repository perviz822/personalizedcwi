import pandas as pd

from wordfreq import zipf_frequency



df=  pd.read_csv('dev.csv')

'''
df['frequency'] = df['Complex_Phrase'].apply(lambda phrase: zipf_frequency(phrase, 'en'))
df.to_csv('test.csv',index=False)

'''

print(len(df))




import nltk
from nltk.corpus import words

# Download the words dataset if not already downloaded
nltk.download('words')

# Load the word list
english_words = set(words.words())

def is_english_word(word):
    return word.lower() in english_words

# Example usage
print(is_english_word("binary"))  # True
print(is_english_word("ceasefire"))  # False



def merge_csv_files(file1, file2, file3):
    """
    Merges three CSV files vertically (appends them together) into a single DataFrame.
    
    Parameters:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        file3 (str): Path to the third CSV file.
    
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    
    merged_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    return merged_df






from wordfreq import zipf_frequency

# Load your DataFrame
df = pd.read_csv('test.csv')

# Create a new column 'frequency' using wordfreq's Zipf frequency
df['length'] = df['Complex_Phrase'].apply(len)

# Save the updated DataFrame
df.to_csv("test.csv", index=False)

# Print the first few rows to verify

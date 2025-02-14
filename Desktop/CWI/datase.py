import pandas as pd

#df = pd.read_csv("hf://datasets/StephanAkkerman/MRC-psycholinguistic-database/mrc_psycholinguistic_database.csv")
#df.to_csv('msc.csv')




# Load the CSV file



import enchant


en_dict = enchant.Dict("en_US")

def is_english_word(word):
    return en_dict.check(word)

# Apply to DataFrame
df = pd.read_csv("dev.csv")
print(len(df))
df = df.dropna(subset=['word'])

df = df[df['word'].apply(is_english_word)]



print(len(df))


df.to_csv('only_english_dev.csv')

print(df)
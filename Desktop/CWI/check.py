import pandas as pd
'''



'''



'''
import pandas as pd

# Read the CSV file (assuming no headers present)





# Save the updated CSV file


'''
'''


'''

'''
df = pd.read_csv("English_Wikipedia_Train.tsv", sep="\t")  # Read TSV
df.to_csv("English_Wikipedia_Train.csv", index=False)  # Save as CSVxs

'''



df =  pd.read_csv('English_Wikipedia_Articles/train.csv')



df.columns = [
    "HIT_ID",
    "Sentence",
    "Start_Offset",
    "End_Offset",
    "Complex_Phrase",
    "Native_Annotators",
    "Non_Native_Annotators",
    "Total_Annotators",
]


#df.to_csv("English_Wikipedia_Articles/train.csv", index=False)


def merge_csv_files(file1,file2,file3,merged_file_name):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    merged_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
    merged_df.to_csv(merged_file_name, index=False)
    pass;
# Save merged data to a new CSV




#merge_csv_files('English_Wikipedia_Articles/test.csv','English_Professional_News/test.csv','English_Wiki_News/test.csv','test.csv')

'''
DROPPING COLUMMNS





df.to_csv('dev.csv',index=False)

'''


'''
df =  pd.read_csv('test.csv')
df["complex_percentage"] = (df["Non_Native_Annotators"] / 10) * 100
df.to_csv('test.csv',index=False

'''

import pandas as pd
from nltk.corpus import stopwords

# Load dataset (example: Yimam et al. 2017 format)
'''




# Define stopwords
stop_words = set(stopwords.words('english'))

# Process phrases and assign complexity
word_complexity = []
for _, row in df.iterrows():
    print(row)
    phrase = row['Complex_Phrase'].lower().split()
    content_words = [word for word in phrase if word not in stop_words]
    for word in content_words:
        word_complexity.append({'word': word, 'complex_percentage': row['complex_percentage']})

# Create new DataFrame
word_df = pd.DataFrame(word_complexity)

# Aggregate by averaging duplicates
word_df = word_df.groupby('word')['complex_percentage'].mean().reset_index()

# Save to CSV
word_df.to_csv("single_word_complexity.csv", index=False)
'''

'''




'''


from wordfreq import zipf_frequency

# Example: Check the Zipf frequency of "boost"



'''
df = pd.read_csv('train.csv')
'''

import pandas as pd

# Load datasets
'''



'''








'''
kuperman = pd.read_csv('msc.csv')
test = pd.read_csv('test.csv') 

# Remove duplicates from Kuperman data (keep first occurrence)
kuperman = kuperman.drop_duplicates(subset=['Word'], keep='first')
kuperman['Word'] = kuperman['Word'].str.lower()

# Merge AoA ratings into train data using left join
# Use 'Word' instead of 'word' for the merge key
test = test.merge(
    kuperman[['Word', 'Familiarity']],  # Ensure column names match
    left_on='word',  # Column in train.csv
    right_on='Word',  # Column in kuperman.csv
    how='left'
)


# Rename the column for clarity


# Optional: Check for words not found in Kuperman dataset


# Save updated train.csv
test.to_csv('test.csv', index=False)





'''







'''
'''




df = pd.read_csv('test.csv')
df = df.drop(df.columns[[0,7]],axis=1)


df.to_csv('test.csv',index=False)









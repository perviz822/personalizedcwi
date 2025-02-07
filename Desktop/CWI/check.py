import pandas as pd
'''



'''



'''
import pandas as pd

# Read the CSV file (assuming no headers present)





# Save the updated CSV file


'''
'''
df =  pd.read_csv('English_News_Dev.csv')
print(len(df))


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


df.to_csv("English_WikiNews_Dev.csv", index=False)

'''

'''


'''
df = pd.read_csv("English_Wikipedia_Train.tsv", sep="\t")  # Read TSV
df.to_csv("English_Wikipedia_Train.csv", index=False)  # Save as CSVxs
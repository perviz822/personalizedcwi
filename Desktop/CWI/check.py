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


df.to_csv("English_Wikipedia_Articles/train.csv", index=False)


def merge_csv_files(file1,file2,file3,merged_file_name):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    merged_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
    merged_df.to_csv(merged_file_name, index=False)
    pass;
# Save merged data to a new CSV


merge_csv_files('English_Wikipedia_Articles/test.csv','English_Professional_News/test.csv','English_Wiki_News/test.csv','test.csv')

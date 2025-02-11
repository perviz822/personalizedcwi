import pandas as pd

#df = pd.read_csv("hf://datasets/StephanAkkerman/MRC-psycholinguistic-database/mrc_psycholinguistic_database.csv")
#df.to_csv('msc.csv')




# Load the CSV file
df = pd.read_csv("msc.csv")

# Filter rows where 'familiarity' is not zero
non_zero_familiarity = df[df["Concretness"] != 0]

# Count the number of such rows


non_zero_familiarity.to_csv('msc_with_familiarity.csv')

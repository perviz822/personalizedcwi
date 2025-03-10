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












'''



'''

df =  pd.read_csv('dev.csv')


import pandas as pd

# Sample DataFrame

# Sort by 'aoa_test_based' in ascending order and drop duplicates based on 'id'
''''

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('your_file.csv')

# Rename the column
df = df.rename(columns={'old_name': 'new_name'})

# Save the DataFrame with the new column name to a new CSV file
df.to_csv('updated_file.csv', index=False)


'''
###############################


'''
df = pd.read_csv('test.csv')

# Rename the column
df = df.rename(columns={'Rating.Mean': 'aoa'})

# Save the DataFrame with the new column name to a new CSV file
df.to_csv('test.csv', index=False)
'''



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV




#filling missing aoa values with logistic regression


train_df = pd.read_csv('train.csv')
validation_df =  pd.read_csv('dev.csv')
test_df =  pd.read_csv('test.csv')

combined_df = pd.concat([
    train_df[train_df['aoa_test_based'].notnull()],
    validation_df[validation_df['aoa_test_based'].notnull()],
    test_df[test_df['aoa_test_based'].notnull()]
])


from sklearn.model_selection import train_test_split

X = combined_df[['length', 'frequency','syllable_count']]  # Features
y = combined_df['aoa_test_based']  # Target (AoA values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Random Forest
'''

param_grid = {
    'n_estimators': [100, 200, 300, 400],  # Number of trees in the forest
    'max_depth': [10, 20, 30, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
}

# Instantiate RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')

# Fit the model
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearchCV
print("Best Hyperparameters:", grid_search.best_params_)


'''




model =RandomForestRegressor(
    bootstrap=False,
    max_depth=30,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=400,
    random_state=42
)
model.fit(X_train, y_train)

# Predict AoA values for the test set
y_pred = model.predict(X_test)


from sklearn.metrics import mean_absolute_error

# Calculate the MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error on Test Set: {mae}')

mean_aoa = y_test.mean()
mae_percentage = (mae / mean_aoa) * 100
print(f'MAE as percentage of mean AoA: {mae_percentage:.2f}%')


# Calculate baseline (mean AoA prediction)
baseline_aoa = combined_df['aoa_test_based'].mean()

# Predicting the mean AoA for all words
baseline_predictions = [baseline_aoa] * len(combined_df)

# Calculate the MAE for the baseline model
baseline_mae = mean_absolute_error(combined_df['aoa_test_based'], baseline_predictions)

# Print the baseline MAE
baseline_mae_percentage = (baseline_mae / mean_aoa) * 100
print(f'MAE as percentage of mean AoA: {baseline_mae_percentage:.2f}%')
import pandas as pd


pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)        # Ensure that all columns fit in the display
pd.set_option('display.max_colwidth', None)

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('results/parameter_testing.csv')

# Display the top 10 models based on train accuracy
print("Top 10 models based on Train Accuracy:")
print(df.nlargest(10, 'train_accuracy'))

print("\n--------------------------------------------------\n")

# Display the top 10 models based on test accuracy
print("Top 10 models based on Test Accuracy:")
print(df.nlargest(10, 'test_accuracy'))
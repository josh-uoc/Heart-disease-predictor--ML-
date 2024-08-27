######################
# IMPORT AND CLEANUP #
######################

# Import necessary libraries
import pandas as pd
import numpy as np
from numpy import random as ra
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings

# Load the data set
df1 = pd.read_csv("Heart disease predictor (ML)\data\Cardiovascular_Disease_Dataset.csv")
print(type(df1))

# Clean the data
df1.dropna(inplace=True)
df1 = df1[df1['serumcholestrol'] != 0]

# Rename columns for clarity
df1 = df1.rename(columns={'patientid': 'Patient Identification Number',
                          'age': 'Age',
                          'gender': 'Gender',
                          'restingBP': 'Resting BP',
                          'serumcholestrol': 'Serum cholesterol',
                          'fastingbloodsugar': 'Fasting blood sugar',
                          'chestpain': 'Chest pain',
                          'restingrelectro': 'Resting EKG result',
                          'maxheartrate': 'Maximum HR achieved',
                          'exerciseangia': 'Exercise induced angina',
                          'oldpeak': 'Oldpeak = ST',
                          'slope': 'Slope of the peak',
                          'noofmajorvessels': 'No. of major vessels',
                          'target': 'Heart disease'})

# Display information about the dataset
df1.info()
df1.describe()
print()


####################
# STARTER ANALYSIS #
####################

# Display value counts for patient ID
print(df1['Patient Identification Number'].value_counts())
print()

# Calculate and display mean age and median serum cholesterol
print("Mean age: ", df1['Age'].mean())
print("Median serum cholesterol: ", df1['Serum cholesterol'].median())
print()

# Calculate and display correlation matrix for selected features
corr_matrix = df1[['Gender', 'Resting BP', 'Resting EKG result', 'No. of major vessels', 'Heart disease']].corr()
print(corr_matrix)
print('\n\n')


##############################
# ANALYSIS AND VISUALISATION #
##############################

# Set up the styling for plots
sns.set_theme(style='darkgrid', palette='bright')
fig, axs = plt.subplots(ncols=4, figsize=(22, 6))

# Create separate DataFrames for presence and absence of heart disease
df_hdabsent = df1[df1['Heart disease'] == 0]
df_hdpresent = df1[df1['Heart disease'] == 1]

# Plot 1: Histogram for BP against heart disease
sns.histplot(data=df_hdpresent, x='Resting BP', color='red', label="Heart disease present", kde=True, binwidth=10, ax=axs[0])
sns.histplot(data=df_hdabsent, x='Resting BP', color='blue', label="Heart disease absent", kde=True, binwidth=10, ax=axs[0])
axs[0].set_title('Resting blood Pressure against heart disease')
axs[0].set_xlabel('Resting BP (mmHg)')
axs[0].set_ylabel('Count')
axs[0].legend()

# Plot 2: Scatterplot for cholesterol and heart disease
sns.stripplot(data=df1, x='Heart disease', y='Serum cholesterol', hue='Heart disease', ax=axs[1])
axs[1].set_title('Serum cholesterol against heart disease')
axs[1].set_xlabel('Presence of heart disease')
axs[1].set_ylabel('Serum cholesterol (mg/dl)')
axs[1].legend(labels=['Heart disease absent', 'Heart disease present'])
axs[1].set_xticks([0, 1])
axs[1].set_xticklabels(['Heart disease absent', 'Heart disease present'])

# Plot 3: Boxplot for vessels blocked and heart disease
sns.boxplot(data=df1, x='Heart disease', y='No. of major vessels', ax=axs[2])
axs[2].set_title('Number of major vessel blockages against heart disease')
axs[2].set_xlabel('Presence of heart disease')
axs[2].set_ylabel('Number of major vessels')
axs[2].set_xticks([0, 1])
axs[2].set_xticklabels(['Absent', 'Present'])


##########################
# MACHINE LEARNING MODEL #
##########################
print("Machine learning model: ")

# Prepare data for the model
columns = ['Resting BP', 'Serum cholesterol', 'No. of major vessels']
X=df1[columns]
y=df1['Heart disease']

print('\nX')
print(X)
print('\ny')
print(y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Scale the features
my_scaler = StandardScaler()
X_train = my_scaler.fit_transform(X_train)
X_test = my_scaler.transform(X_test)

# Train the logistic regression model
my_regressor = LogisticRegression()
my_regressor.fit(X_train, y_train)

# Make predictions and print classification report
predictions = my_regressor.predict(X_test)
report = classification_report(y_test, predictions)
print(report)

# Make a prediction for a random sample
ran_index = ra.randint(0, len(X_test))
X_test_ran = X_test[ran_index].reshape(-1, 3)
print('Random index: ', ran_index)
print()

ran_prediction = my_regressor.predict(X_test_ran)
ran_probs = my_regressor.predict_proba(X_test_ran)

print('Prediction: ', ran_prediction[0])
print('Probabilities: ', ran_probs)

for x in ran_probs:
  for y in x:
    print(f'{round(y*100)}%')
print()


# Get user input for prediction
warnings.filterwarnings(action="ignore", category=UserWarning, module="sklearn.base")
resting_bp = float(input("Enter patient resting BP: "))
serum_chol = float(input("Enter patient serum cholesterol: "))
vessels = int(input("Enter patient number of major vessel blockages (0-3): "))

# Prepare user input for prediction
X_array2 = np.array([resting_bp,
                    serum_chol,
                    vessels]).reshape(-1, 3)
print(X_array2)

# Scale user input and make prediction
X_array2_scaled = my_scaler.transform(X_array2)
new_prediction = my_regressor.predict(X_array2_scaled)
new_probs = my_regressor.predict_proba(X_array2_scaled)

print("Presence of heart disease: ", new_prediction[0])
print("Probability: ", new_probs)
print()

# Create DataFrame for user input
df_newX = pd.DataFrame(X_array2,
                      columns = ['Resting BP', 'Serum cholesterol', 'Number of major blockages'])
print(df_newX)
print()

# Determine colour based on prediction
match new_prediction[0]:
  case 'Absence of heart disease':
    c = 'blue'
  case 'Presence of heart disease':
    c = 'orangered'

# Plot 4: Pie chart for probability of heart disease
sns.set_theme(style="darkgrid", palette="pastel")
axs[3].pie(new_probs[0], labels=['Absence', 'Presence'], colors=['blue', 'orangered'], autopct='%1.1f%%')
axs[3].set_title("Probability of heart disease")

# Finalize and save the plot
plt.tight_layout()
plt.savefig("Heart disease predictor (ML)\img")
plt.show()










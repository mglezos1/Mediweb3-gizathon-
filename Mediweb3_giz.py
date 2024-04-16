import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Step 1: Read medical data from CSV file made from the data entry application
df = pd.read_csv('MedicalData.csv')

# Count the number of positive virus cases from Company A
num_positive_cases_A = df[(df['Virus'] == 'Positive') & (df['Company Test'] == 'A')].shape[0]

# Print out the number of positive cases from Company A
print("Number of positive cases:", num_positive_cases_A)

# Step 2: Filter the dataset to only include the relevant information
filtered_df = df[(df['Virus'] == 'Positive') & (df['Company Test'] == 'A')]

# Step 3: Prepare data
X = filtered_df[['Weight']].copy()  # Using Weight as the only feature
y = filtered_df['Virus'].copy()  # Virus status as label

# Encode categorical variables
encoder_X = LabelEncoder()
encoder_y = LabelEncoder()

X_encoded = encoder_X.fit_transform(X['Weight'])
y_encoded = encoder_y.fit_transform(y)

# Use the encoded labels for both training and test data
X.loc[:, 'Weight'] = X_encoded  # Use all weights for training data
y = y_encoded

# Step 4: Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Print out the results
print("\nDecision Tree Classifier trained successfully.")

# Step 5: Convert the trained model to ONNX format
initial_type = [('float_input', FloatTensorType([None, 1]))]  # Define input type
onnx_model = convert_sklearn(clf, initial_types=initial_type)

# Step 6: Save the ONNX model to a file
with open("decision_tree.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved successfully.")

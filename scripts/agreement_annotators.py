#to calc the agreement between annotators

from sklearn.metrics import cohen_kappa_score
import itertools
import pandas as pd
import json
import os
from nltk.metrics import agreement


# Navigate to the parent directory as global path
script_directory = os.path.dirname(os.path.abspath(__file__)) 
current_path = os.path.dirname(script_directory)
global_path = os.path.dirname(current_path) #go to the previous level for the parent
print(global_path)
print('Global path: ', global_path)
saved_result_path = os.path.join(global_path, "saved_results")  # Path to the saved_results folder
print(print('Saving path: ', saved_result_path))

# Load the three DataFrames
angelo_df = pd.read_csv('/home/saram/PhD/Proximity_AI/database/labeled/agereement_conll/agreement_Angelo.csv')
sara_df = pd.read_csv('/home/saram/PhD/Proximity_AI/database/labeled/agereement_conll/agreement_Sara.csv')
daniela_df = pd.read_csv('/home/saram/PhD/Proximity_AI/database/labeled/agereement_conll/agreement_Daniela.csv')

# Extract common patients (IDs)
common_patients = set(angelo_df['ID']).intersection(sara_df['ID']).intersection(daniela_df['ID'])

# Print the number of common patients
print(f"Number of common patients: {len(common_patients)}")

# Update each DataFrame to contain only common patients
angelo_df = angelo_df[angelo_df['ID'].isin(common_patients)]
sara_df = sara_df[sara_df['ID'].isin(common_patients)]
daniela_df = daniela_df[daniela_df['ID'].isin(common_patients)]

# Print the updated DataFrames
print("\nUpdated Angelo DataFrame:")
print(angelo_df.head(5))

print("\nUpdated Sara DataFrame:")
print(sara_df.head(5))

print("\nUpdated Daniela DataFrame:")
print(daniela_df.head(5))



# List of annotators
names = ['Angelo', 'Sara', 'Daniela']

# Folder to save results
output_folder = 'saved_results'#/home/saram/PhD/Proximity_AI/database/labeled/agereement_conll'
os.makedirs(output_folder, exist_ok=True)

# Process each annotator
for name in names:
    print(f'elaborating df of {name}')
    if 'Angelo' in name:
        file_path = '/home/saram/PhD/Proximity_AI/database/labeled/agereement_conll/agreement_Angelo.csv'
    elif 'Sara' in name:
        file_path = '/home/saram/PhD/Proximity_AI/database/labeled/agereement_conll/agreement_Sara.csv'
    else:
        file_path = '/home/saram/PhD/Proximity_AI/database/labeled/agereement_conll/agreement_Daniela.csv'


    annotator_df = pd.read_csv(file_path)
    print(annotator_df.columns)

    # Initialize lists to store parsed information
    id_list = []
    start_list = []
    end_list = []
    text_list = []
    label_list = []

    # Iterate through each row in the dataframe
    for index, row in annotator_df.iterrows():
        patient_id = row['ID']
        # Check for NaN values in the 'label' column
        if pd.notna(row['label']):
            annotations = json.loads(row['label'])
            #print('printing the label', row['label'])
            annotations = json.loads(row['label'])
            
            # Iterate through each annotation for the current patient
            for annotation in annotations:
                id_list.append(patient_id)
                start_list.append(annotation['start'])
                end_list.append(annotation['end'])
                text_list.append(annotation['text'])
                label_list.append(annotation['labels'][0])

    # Create a new DataFrame from the parsed information
    new_df = pd.DataFrame({
        'ID': id_list,
        'Start': start_list,
        'End': end_list,
        'Text': text_list,
        'Label': label_list
    })

    # Save the new DataFrame to a CSV file
    output_file_path = os.path.join(output_folder, f'{name}_annotations.csv')
    new_df.to_csv(output_file_path, index=False)

    # Print the new DataFrame with annotations
    print(f"\nNew DataFrame with Annotations for {name}:")
    #print(new_df)
    print(f"\nSaved results to: {output_file_path}")

# Load the three DataFrames
file_path ='/home/saram/PhD/Proximity_AI/saved_results/Angelo_annotations.csv'# '/home/saram/PhD/Proximity_AI/database/labeled/agreement_conll/Angelo_annotations.csv'#'/home/saram/PhD/Proximity_AI/database/labeled/agereement_conll/agreement_Angelo.csv'
angelo_df = pd.read_csv(file_path)

sara_df = pd.read_csv('/home/saram/PhD/Proximity_AI/saved_results/Sara_annotations.csv')

daniela_df = pd.read_csv('/home/saram/PhD/Proximity_AI/saved_results/Daniela_annotations.csv')


# Extract common patients (IDs)
common_patients = set(angelo_df['ID']).intersection(sara_df['ID']).intersection(daniela_df['ID'])

# Print the number of common patients
print(f"Number of common patients: {len(common_patients)}")

# Update each DataFrame to contain only common patients
angelo_df = angelo_df[angelo_df['ID'].isin(common_patients)]
sara_df = sara_df[sara_df['ID'].isin(common_patients)]
daniela_df = daniela_df[daniela_df['ID'].isin(common_patients)]

print('Angelo: ', angelo_df)
print(angelo_df.shape)

print('Sara: ', sara_df)
print(sara_df.shape)

print('Daniela: ', daniela_df)
print(daniela_df.shape)


# Function to calculate Cohen's Kappa between two annotators
def calculate_kappa(df1, df2):
    combined_df = pd.merge(df1, df2, on=['ID', 'Start', 'End'], how='outer', suffixes=('_1', '_2'))
    combined_df = combined_df.fillna('')
    
    labels_1 = combined_df['Label_1'].tolist()
    labels_2 = combined_df['Label_2'].tolist()

    kappa = cohen_kappa_score(labels_1, labels_2)
    return kappa



# Function to calculate Overall Accuracy
def calculate_accuracy(df1, df2):
    combined_df = pd.merge(df1, df2, on=['ID', 'Start', 'End'], how='outer', suffixes=('_1', '_2'))
    combined_df = combined_df.fillna('')
    
    correct_labels = combined_df[combined_df['Label_1'] == combined_df['Label_2']]
    accuracy = len(correct_labels) / len(combined_df)
    return accuracy


# Function to calculate Inter-Rater Agreement (IRA)
def calculate_ira(df_list):
    total_annotations = 0
    agreed_annotations = 0

    for i in range(len(df_list)):
        for j in range(i + 1, len(df_list)):
            annotator1_data = df_list[i]
            annotator2_data = df_list[j]

            # Find common columns for comparison
            common_columns = list(set(annotator1_data.columns) & set(annotator2_data.columns))
            
            # Merge dataframes on common columns
            merged_df = pd.merge(annotator1_data, annotator2_data, on=common_columns, suffixes=('_1', '_2'))
            print(merged_df.columns)
            
            total_annotations += len(merged_df)
            agreed_annotations += sum(merged_df[common_columns[0] + '_1'] == merged_df[common_columns[0] + '_2'])

    ira = agreed_annotations / total_annotations
    return ira

# Function to calculate Cohen's Kappa by category
def calculate_kappa_by_category(df_list):
    category_kappas = {}

    for category in df_list[0]['Label'].unique():
        category_annotations = []

        for df in df_list:
            category_annotations.append(df[df['Label'] == category]['Label'].tolist())

        # Calculate Cohen's Kappa for the current category
        kappa = cohen_kappa_score(category_annotations[0], category_annotations[1])
        category_kappas[category] = kappa

    return category_kappas

# Calculate Cohen's Kappa for each pair of annotators
pairs = list(itertools.combinations(['angelo', 'sara', 'daniela'], 2))
for pair in pairs:
    annotator1 = pair[0]
    annotator2 = pair[1]

    kappa = calculate_kappa(globals()[f'{annotator1}_df'], globals()[f'{annotator2}_df'])
    print(f"Cohen's Kappa between {annotator1} and {annotator2}: {kappa}")

# Calculate Overall Accuracy
accuracy_angelo_sara = calculate_accuracy(angelo_df, sara_df)
accuracy_angelo_daniela = calculate_accuracy(angelo_df, daniela_df)
accuracy_sara_daniela = calculate_accuracy(sara_df, daniela_df)

print(f"Overall Accuracy between Angelo and Sara: {accuracy_angelo_sara}")
print(f"Overall Accuracy between Angelo and Daniela: {accuracy_angelo_daniela}")
print(f"Overall Accuracy between Sara and Daniela: {accuracy_sara_daniela}")

# Calculate Inter-Rater Agreement (IRA)
#ira = calculate_ira([angelo_df, sara_df, daniela_df])
#print(f"Inter-Rater Agreement (IRA): {ira}")

# Calculate Cohen's Kappa by category
category_kappas = calculate_kappa_by_category([angelo_df, sara_df, daniela_df])
print("\nCohen's Kappa by Category:")
for category, kappa in category_kappas.items():
    print(f"{category}: {kappa}")

'''
# Preprocess data if needed
# ...

# Define a function to calculate Cohen's Kappa
def calculate_kappa(annotator1, annotator2):
    # Implement the calculation here
    return cohen_kappa_score(annotator1, annotator2)

# Calculate pairwise agreement
sara_angelo_agreement = calculate_kappa(sara_df['annotations'], angelo_df['annotations'])
sara_daniela_agreement = calculate_kappa(sara_df['annotations'], daniela_df['annotations'])
angelo_daniela_agreement = calculate_kappa(angelo_df['annotations'], daniela_df['annotations'])

# Calculate overall agreement
overall_agreement = (sara_angelo_agreement + sara_daniela_agreement + angelo_daniela_agreement) / 3

# Report the results
print(f'Sara vs. Angelo Agreement: {sara_angelo_agreement}')
print(f'Sara vs. Daniela Agreement: {sara_daniela_agreement}')
print(f'Angelo vs. Daniela Agreement: {angelo_daniela_agreement}')
print(f'Overall Agreement: {overall_agreement}')
'''

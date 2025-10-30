import pandas as pd

CANCER_FILE_FP = 'RawData/Cancer_Data.csv'
DATASET_FP = 'RawData/dataset.csv'
SYMPTOM_DESCRIPTION_FP = 'RawData/symptom_Description.csv'
SYMPTOM_PRECAUTION_FP = 'RawData/symptom_precaution.csv'
SYMPTOM_SEVERITY_FP = 'RawData/Symptom-severity.csv'

def load_data_as_df(file_path=DATASET_FP):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        print(f"DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

if __name__ == "__main__":
    df = load_data_as_df()
    print(df)
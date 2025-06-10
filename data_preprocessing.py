import pandas as pd
import numpy as np
import chardet

def get_processed_data(new_csv_path):
    # Detect encoding of the new CSV
    with open(new_csv_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']

    # Load the new comprehensive_cosmetics_export_analysis.csv
    df = pd.read_csv(new_csv_path, encoding=encoding)

    # Select and rename columns based on the new CSV structure
    df_processed = df[[
        'Country_Korean',
        '2024_Export_1000USD',
        '2024_Growth_Rate',
        'Risk_Index',
        'Safety_Focused_Score',
        'Growth_Focused_Score',
        'Export_Focused_Score',
        'Balanced_Score'
    ]].copy()

    df_processed.columns = [
        'Country',
        'Export_Value',
        'Growth_Rate',
        'Risk_Score',
        'Safety_Score',
        'Growth_Score',
        'Export_Score',
        'Balanced_Score'
    ]

    # Filter out any rows that might have NaN in critical columns if necessary
    df_processed = df_processed.dropna(subset=['Country', 'Export_Value', 'Growth_Rate', 'Risk_Score'])

    return df_processed

# This part is for testing the script independently, not for Streamlit app
if __name__ == '__main__':
    new_csv_file = '/home/ubuntu/upload/comprehensive_cosmetics_export_analysis.csv'
    processed_df = get_processed_data(new_csv_file)
    processed_df.to_csv('/home/ubuntu/processed_2024_data.csv', index=False)
    print("Data preprocessing complete. processed_2024_data.csv created for testing.")

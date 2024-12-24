from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, logger
import pandas as pd
import pyreadstat
import argparse
import os

def load_data(filename):
    """Load data from a file based on its extension."""
    file_path = RAW_DATA_DIR / filename
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension == '.csv':
        logger.info(f"Loading CSV file from {file_path}")
        return pd.read_csv(file_path)
    elif file_extension == '.sas7bdat':
        logger.info(f"Loading SAS file from {file_path}")
        df, meta = pyreadstat.read_sas7bdat(file_path)
        return df
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def save_data(df, filename):
    """Save data to a file."""
    df.to_csv(PROCESSED_DATA_DIR / filename, index=False)
    
def process_dataset(input_filename, output_filename):
    # Construct the full path for the input and output files
    input_path = RAW_DATA_DIR / input_filename
    output_path = PROCESSED_DATA_DIR / output_filename

    # Load the dataset
    logger.info(f"Loading dataset from {input_path}")
    df, meta = pyreadstat.read_sas7bdat(input_path)

    # Keep only the specified columns
    columns_to_keep = ['UEN', 'ONLINE_PRESENCE_IND', 'CORPORATE_URL_IND', 'ECOM_REV', 'ECOM_REV_IND']
    df = df[columns_to_keep]

    # Filter records where ONLINE_PRESENCE_IND is 1
    df = df[df['ONLINE_PRESENCE_IND'] == 1]

    # Generate the new column 'TRUE_IE'
    df['TRUE_IE'] = df.apply(
        lambda row: 'C' if row['ECOM_REV'] and row['ECOM_REV_IND'] == 1 else 'B', axis=1
    )
    
    # fix UEN col. 
    def extract_uen(uen):
        return uen.split('-')[0]
    df['UEN'] = df['UEN'].apply(extract_uen)
    
    # Save the final dataset to the processed data directory
    logger.info(f"Saving processed dataset to {output_path}")
    df.to_csv(output_path, index=False)
    
    return df

def find_unmatched(df, col1, col2, output_csv):
    """
    This function returns a new DataFrame with rows where the values in col1 and col2 do not match.
    It also saves this new DataFrame to a CSV file.
    """
    
    # Find rows where col1 and col2 values do not match
    unmatched_df = df[df[col1] != df[col2]]
    
    # Save the new DataFrame to a CSV file
    output_path = PROCESSED_DATA_DIR / output_csv
    unmatched_df.to_csv(output_path, index=False)
    
    return unmatched_df

def imputation_v1(out2022, out2023):
    
    # import datasets
    soe_path = PROCESSED_DATA_DIR / 'soe2022_labelled.csv'
    df2022_path = RAW_DATA_DIR / 'ie2022.csv'
    df2023_path = RAW_DATA_DIR / 'ie2023.csv'
    soe_df = pd.read_csv(soe_path)
    df2022 = pd.read_csv(df2022_path)
    df2023 = pd.read_csv(df2023_path)
    
    # Create a dictionary from soe_df for quick lookup
    uen_to_true_ie = dict(zip(soe_df['UEN'], soe_df['TRUE_IE']))
    
    # Function to impute 'FINAL_IE' based on 'UEN'
    def impute(row):
        return uen_to_true_ie.get(row['UEN'], row['FINAL_IE'])
    
    # Apply the impute function to 'FINAL_IE' column of df2022 and df2023
    df2022['FINAL_IE'] = df2022.apply(impute, axis=1)
    df2023['FINAL_IE'] = df2023.apply(impute, axis=1)
    
    # Save outputs
    output_path2022 = PROCESSED_DATA_DIR / out2022
    output_path2023 = PROCESSED_DATA_DIR / out2023
    df2022.to_csv(output_path2022, index=False)
    df2023.to_csv(output_path2023, index=False)
    
    return df2022, df2023

import pandas as pd

def create_train_test_sets():
    """
    Create training and test sets by merging with keyword_2022 on 'UEN'.

    Parameters:
    soe_data (pd.DataFrame): Source data for training set
    ie2022 (pd.DataFrame): Source data for test set (part 1)
    ie2023 (pd.DataFrame): Source data for test set (part 2)
    keyword_2022 (pd.DataFrame): Data to be merged with on 'UEN'

    Returns:
    train_set (pd.DataFrame): Resulting training set
    test_set (pd.DataFrame): Resulting test set
    """
    # Get paths
    soe_path = PROCESSED_DATA_DIR / 'soe2022_labelledv1.csv'
    test2022_path = RAW_DATA_DIR / 'ie2022.csv'
    keyword2022_path = RAW_DATA_DIR / 'IE2022_keywords170k.csv'
    keyword2023_path = RAW_DATA_DIR / 'IE2023_keywords200k.csv'
    
    # import dataframes
    soe_data = pd.read_csv(soe_path)
    ie2022 = pd.read_csv(test2022_path)
    keyword_2022 = pd.read_csv(keyword2022_path)
    keyword_2023 = pd.read_csv(keyword2023_path)
    
    # Subset 'UEN' and merge for train set
    # Train set expanded by first merging with keyword2022, then keyword2023 (remaining records)
    # First merge with keyword_2022
    soe_data = soe_data[['UEN', 'TRUE_IE']]
    merged_2022 = pd.merge(soe_data, keyword_2022, on='UEN', how='inner')

    # Find remaining records in soe_data that were not merged
    remaining_soe_data = soe_data[~soe_data['UEN'].isin(merged_2022['UEN'])]

    # Merge remaining records with keyword_2023
    merged_2023 = pd.merge(remaining_soe_data, keyword_2023, on='UEN', how='inner')

    # Concatenate both results
    train_set = pd.concat([merged_2022, merged_2023], ignore_index=True)
    
    # Create test set for IE2022 data - Subset 'UEN' and merge for 2022 test set
    test_set_2022 = pd.merge(ie2022[['UEN']], keyword_2022, on='UEN', how='inner')
    
    # Drop unnecc cols
    dataframes = [train_set, test_set_2022]
    columns_to_drop =['Index', 'SRC_NAME', 'ENTP_NM', 'ST', 'CURR_SSIC', 'LIVE_IND', 'Company UEN', 'Website', 'F1', 'NAME', 'WEBSITE', 'SOURCE', 'ST_EFF_DT', 'INDEX']
    for df in dataframes:
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
    # Re-create & use 'sum' column (seems to be buggy)
    # Function to drop the 'sum' column and create a new 'sum' column
    def process_dataset(df):
        if 'sum' in df.columns:
            df = df.drop(columns=['sum'])
        df['sum'] = df.select_dtypes(include='number').sum(axis=1)
        return df

    # Process both datasets
    train_set = process_dataset(train_set)
    test_set_2022 = process_dataset(test_set_2022)
        
    # save train & test set
    train_path = PROCESSED_DATA_DIR / 'train.csv'
    test_path = PROCESSED_DATA_DIR / 'test2022.csv'
    train_set.to_csv(train_path, index=False)
    test_set_2022.to_csv(test_path, index=False)
    
    return train_set, test_set_2022

def merge_inference_with_indicators(): 
    # Read the datasets
    ie2022_path = RAW_DATA_DIR / 'ie2022.csv'
    inference_path = PROCESSED_DATA_DIR / 'ie2022_inferencev2.csv'
    df = pd.read_csv(ie2022_path)
    df_inference = pd.read_csv(inference_path)
    
    # Inference UEN with "B" and "C"
    uen_b = df_inference.loc[df_inference["pred_class"] == 'B']["UEN"]
    uen_c = df_inference.loc[df_inference["pred_class"] == 'C']["UEN"]
    
    # Imputation step - adjust "FINAL_IE" values accordingly
    df.loc[(df['UEN'].isin(uen_b)) & (df["FINAL_IE"].isin(['B1', 'B2', 'C1', 'C2'])), 'FINAL_IE'] = 'B'
    df.loc[(df['UEN'].isin(uen_c)) & (df["FINAL_IE"].isin(['B1', 'B2', 'C1', 'C2'])), 'FINAL_IE'] = 'C'
    
    # Check & convert ("B" & "C")
    df['FINAL_IE'] = df['FINAL_IE'].replace({'B1': 'B', 'B2': 'B', 'C1': 'C', 'C2': 'C'})
    
    # return output
    output_path = PROCESSED_DATA_DIR / 'ie2022_updatedv3.csv'
    df.to_csv(output_path, index=False)
    
    return df

# def main():
#     parser = argparse.ArgumentParser(description='Data processing script.')
#     subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

#     # Sub-parser for the load_data function
#     parser_load = subparsers.add_parser('load', help='Load data from a file')
#     parser_load.add_argument('input_filename', type=str, help='The input filename')

#     # Sub-parser for the save_data function
#     parser_save = subparsers.add_parser('save', help='Save data to a file')
#     parser_save.add_argument('input_filename', type=str, help='The input filename')
#     parser_save.add_argument('output_filename', type=str, help='The output filename')

#     # Sub-parser for the process_dataset function
#     parser_process = subparsers.add_parser('process', help='Process a dataset')
#     parser_process.add_argument('input_filename', type=str, help='The input filename')
#     parser_process.add_argument('output_filename', type=str, help='The output filename')
    
#     args = parser.parse_args()
    
#     if args.command == 'load':
#         df = load_data(args.input_filename)
#         print(df.head())  # Print the first few rows to verify the load operation
#     elif args.command == 'save':
#         df = load_data(args.input_filename)
#         save_data(df, args.output_filename)
#     elif args.command == 'process':
#         process_dataset(args.input_filename, args.output_filename)

if __name__ == "__main__":
    main()
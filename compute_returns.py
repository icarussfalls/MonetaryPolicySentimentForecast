import pandas as pd
from datetime import datetime, timedelta

# Load published dates
pub_dates = pd.read_csv('published_dates.csv')


# Load all index CSVs from index_data folder

import glob
import os
index_folder = 'index_data'
index_files = glob.glob(os.path.join(index_folder, '*.csv'))

index_data_list = []

for f in index_files:
    df = pd.read_csv(f)
    # Try to find the date column
    date_col = None
    for col in df.columns:
        if col.lower() in ['timestamp', 'date', 'datetime']:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"No date column found in {f}. Columns: {df.columns}")
    df['Date'] = pd.to_datetime(df[date_col])
    # Try to infer index name from filename or column
    if 'IndexName' in df.columns:
        idx_name = df['IndexName'].iloc[0]
    else:
        idx_name = os.path.splitext(os.path.basename(f))[0]
    df['IndexName'] = idx_name
    index_data_list.append(df)

index_data = pd.concat(index_data_list, ignore_index=True)

results = []

# Fix column name: strip spaces and check for correct column
pub_dates.columns = pub_dates.columns.str.strip()
if 'PublishedDate' not in pub_dates.columns:
    # Try to find a similar column
    for col in pub_dates.columns:
        if 'publish' in col.lower():
            pub_dates.rename(columns={col: 'PublishedDate'}, inplace=True)
            break

for _, row in pub_dates.iterrows():
    fy = row['FY']
    pub_date = pd.to_datetime(row['PublishedDate'], dayfirst=True)
    date_30 = pub_date + timedelta(days=30)
    
    # For each index, compute return
    for idx in index_data['IndexName'].unique():
        idx_df = index_data[index_data['IndexName'] == idx]
        # Get close price at published date (or closest after)
        start_val = idx_df[idx_df['Date'] >= pub_date].sort_values('Date').iloc[0]['close']
        # Get close price at 7 days after (or closest after)
        end_val = idx_df[idx_df['Date'] >= date_30].sort_values('Date').iloc[0]['close']
        ret = (end_val - start_val) / start_val
        results.append({
            'FY': fy,
            'IndexName': idx,
            'PublishedDate': pub_date.strftime('%Y-%m-%d'),
            'Date_30': date_30.strftime('%Y-%m-%d'),
            'Return_30d': ret
        })

returns_df = pd.DataFrame(results)
returns_df.to_csv('index_30d_returns.csv', index=False)
print('Saved index_30d_returns.csv')

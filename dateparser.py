import pandas as pd
from dateutil import parser

# Load your FY/PublishedDate CSV
df = pd.read_csv("published_dates.csv")  # replace with your actual path

# Convert to datetime
def safe_parse(date_str):
    try:
        return parser.parse(str(date_str), fuzzy=True, dayfirst=False)
    except:
        return pd.NaT

df['PublishedDatetime'] = df['PublishedDate'].apply(safe_parse)

# Extract year from the parsed datetime
df['Year'] = df['PublishedDatetime'].dt.year

# Extract FY start year (e.g., from "2020-21")
df['FY_Year'] = df['FY'].apply(lambda x: int(x.split("-")[0]))

print(df[['FY', 'PublishedDate', 'PublishedDatetime', 'Year', 'FY_Year']])

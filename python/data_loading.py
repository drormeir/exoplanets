import warnings
import os
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from python.display_utils import display_with_title
import matplotlib.pyplot as plt

def load_exoplanet_data(project_path: str, name: str = 'pscomppars', display_head: int|None = None):
    filename = f"{name}.csv" if not name.endswith('.csv') else name
    filename = os.path.join(project_path, 'datasets', filename)
    if not os.path.exists(filename):
        print("Downloading pscomppars.csv ... This may take a while...")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=".*has a unit but is kept as a MaskedColumn.*", category=UserWarning)       
            # Query the NASA Exoplanet Archive
            table = NasaExoplanetArchive.query_criteria(table='pscomppars')
        # List of columns known to have logarithmic (dex) units that cause warnings
        dex_columns = ['st_lum', 'st_lumerr1', 'st_lumerr2', 'st_logg', 'st_loggerr1', 'st_loggerr2']        
        for col in dex_columns:
            if col in table.colnames:
                # Extract the raw data without units
                table[col] = table[col].data
                print(f"Processed column {col} to remove unit conversion warnings")
        table.write(filename, format='csv', overwrite=True)

    exoplanet_data = pd.read_csv(filename)

    print(f"Dataset {name} loaded with shape: {exoplanet_data.shape}")
    if display_head is not None:
        display_with_title(exoplanet_data.head(display_head), f"First {display_head} rows of the {name} dataset")

    return exoplanet_data



def set_index_remove_null_columns(df : pd.DataFrame, verbose : bool = True):
    null_rows = df.isnull().all(axis=1)
    df = df[~null_rows]
    nrows = df.shape[0]
    dtypes = df.dtypes.copy()
    dtypes.name = 'Data Type'
    dtypes.index.name = 'Column Name'

    nunique_counts = df.nunique()
    nunique_counts.name = 'Unique Count'
    nunique_counts.reindex(dtypes.index, fill_value=0)
    nunique_counts[nunique_counts <= 1] = 0

    null_counts_per_column = df.isnull().sum()
    null_counts_per_column.name = 'Null Count'
    null_counts_per_column.reindex(dtypes.index, fill_value=0)
    null_counts_per_column[nunique_counts <= 1] = nrows

    columns_statistics = pd.concat([dtypes, null_counts_per_column, nunique_counts], axis=1)
    columns_statistics.columns = ['Data Type', 'Null Count', 'Unique Count']

    dtypes_counts = dtypes.value_counts()
    is_fully_unique = columns_statistics['Unique Count'] == nrows
    if df.index.name is None and df.index.equals(pd.RangeIndex(start=0, stop=len(df), step=1)):
        # DataFrame uses the default integer index --> check it can be replaced by a column
        can_be_index = columns_statistics[is_fully_unique & (columns_statistics['Data Type'] != 'float64')].index.tolist()
        if not can_be_index:
            print("No single column that can be used as index")
        elif len(can_be_index) == 1:
            print(f"Setting {can_be_index[0]} as index")
            df.set_index(can_be_index[0], inplace=True)
            assert can_be_index[0] not in df.columns, f"{can_be_index[0]} is in the columns"
            return set_index_remove_null_columns(df, verbose=verbose)
        else:
            print("Multiple columns that can be used as index:", can_be_index)

    is_high_null_columns = columns_statistics['Null Count'] >= nrows // 2
    if is_high_null_columns.any():
        print("\nDropping columns with only null values or high ratio of nulls:\n" + "-"*60)
        null_columns = columns_statistics[is_high_null_columns]
        for dtype, dtype_df in null_columns.groupby('Data Type'):
            null_columns_list = dtype_df.index.tolist()
            print(f"There are {len(null_columns_list)} columns of type {dtype} with only null values:\n{null_columns_list}\n" + "-"*60)
        df.drop(null_columns.index.tolist(), axis=1, inplace=True)
        return set_index_remove_null_columns(df, verbose=verbose)


    no_nulls_counts = columns_statistics[columns_statistics['Null Count'] == 0]['Data Type'].value_counts()
    no_nulls_counts = no_nulls_counts.reindex(dtypes_counts.index, fill_value=0)
    
    fully_unique_counts = columns_statistics[is_fully_unique]['Data Type'].value_counts()
    fully_unique_counts = fully_unique_counts.reindex(dtypes_counts.index, fill_value=0)

    rest_counts = dtypes_counts - fully_unique_counts - no_nulls_counts
    feature_counts = pd.concat([no_nulls_counts, fully_unique_counts, rest_counts], axis=1)
    feature_counts.columns = ['Features Without Nulls', 'Features Fully Unique', 'Rest']
    feature_counts.index.name = 'Data Type'
    feature_counts.fillna(0, inplace=True)
    feature_counts['Total per dtype'] = feature_counts.sum(axis=1)
    feature_counts.loc['Total'] = feature_counts.sum(axis=0)



    if verbose:
        display_with_title(columns_statistics, "Columns Statistics Summary")
        display_with_title(feature_counts, "Feature Counts Summary")
        if plt.get_backend() == 'agg':
            print("Warning: FigureCanvasAgg is non-interactive, and thus cannot be shown")
        else:
            null_counts_per_column_percent = null_counts_per_column.sort_values(ascending=False) / nrows
            plt.figure(figsize=(10, 6))
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%'))
            null_counts_per_column_percent.plot(kind='bar', color='skyblue')
            plt.title(f'Histogram of nulls per column (Percentage)')
            plt.xlabel('Column Name')
            plt.ylabel('Percentage of Nulls per Column')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            df.isnull().sum(axis=1).plot(kind='hist', bins=30, color='salmon')
            plt.title(f'Histogram of Null Values per Row (row length = {df.shape[1]})')
            plt.xlabel('Number of Nulls per Row')
            plt.ylabel('Number of Rows')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            nunique_counts.sort_values(ascending=False).plot(kind='bar', color='lightgreen')
            plt.title(f'Histogram of Unique Features per Column (total rows: {nrows})')
            plt.xlabel('Column Name')
            plt.ylabel('Number of Unique Features per Column')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
        print(f'Final dataset shape: {df.shape}')
    return columns_statistics, feature_counts


if __name__ == "__main__":
    exoplanet_data = load_exoplanet_data(display_head=5)
    columns_statistics, feature_counts = set_index_remove_null_columns(exoplanet_data)

import os
import pandas as pd
import re
from transformers import pipeline
from prophet import Prophet

# Set the environment variable for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def list_excel_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.xlsx') or f.endswith('.xlsb')]

def select_file(files):
    print("Select a file to analyze:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    choice = int(input("Enter the number of the file: ")) - 1
    return files[choice]

def select_sheet(excel_file):
    sheets = pd.ExcelFile(excel_file).sheet_names
    print("Select a sheet to analyze:")
    for i, sheet in enumerate(sheets):
        print(f"{i + 1}. {sheet}")
    choice = int(input("Enter the number of the sheet: ")) - 1
    return sheets[choice]

def recommend_columns(headers):
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    labels = ['date', 'manufacturer', 'item number']

    results = {label: [] for label in labels}

    for header in headers:
        result = classifier(header, candidate_labels=labels)
        for label in labels:
            score = result['scores'][result['labels'].index(label)]
            results[label].append((header, score))

    ranked_columns = {label: sorted(columns, key=lambda x: x[1], reverse=True) for label, columns in results.items()}

    return ranked_columns

def select_column(ranked_columns, column_type):
    print(f"Select a {column_type} column:")
    for i, (header, score) in enumerate(ranked_columns[column_type]):
        print(f"{i + 1}. {header} (Score: {score:.2f})")
    choice = int(input(f"Enter the number of the {column_type} column: ")) - 1
    return ranked_columns[column_type][choice][0]

def convert_excel_date(date_series):
    # Handle both datetime and numeric date formats
    if pd.api.types.is_numeric_dtype(date_series):
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(date_series, 'D')
    else:
        return pd.to_datetime(date_series, errors='coerce')

def forecast_usage(data, date_column, manufacturer_column, item_column, top_items):
    # Prepare the dataframe for forecasting
    forecast_results = []

    for manufacturer, item in top_items:
        item_data = data[(data[manufacturer_column] == manufacturer) & (data[item_column] == item)]
        item_data = item_data[[date_column]].copy()
        item_data['y'] = item_data.groupby(date_column)[date_column].transform('count')
        item_data = item_data.drop_duplicates().rename(columns={date_column: 'ds'})

        if len(item_data) < 2:
            continue  # Skip items with less than 2 data points

        # Initialize Prophet model
        model = Prophet()
        model.fit(item_data)

        # Create future dataframe
        future = model.make_future_dataframe(periods=52, freq='W')

        # Make forecast
        forecast = model.predict(future)
        forecast['manufacturer'] = manufacturer
        forecast['item'] = item
        forecast_results.append(forecast[['ds', 'yhat', 'manufacturer', 'item']])

    # Combine all forecast results
    combined_forecast = pd.concat(forecast_results, ignore_index=True)

    return combined_forecast

def main():
    directory = '.'  # Change this to your target directory
    files = list_excel_files(directory)
    
    if not files:
        print("No Excel files found in the directory.")
        return

    excel_file = os.path.join(directory, select_file(files))
    sheet_name = select_sheet(excel_file)

    data = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl' if excel_file.endswith('.xlsx') else 'pyxlsb')
    headers = data.columns.tolist()

    ranked_columns = recommend_columns(headers)

    date_column = select_column(ranked_columns, 'date')
    manufacturer_column = select_column(ranked_columns, 'manufacturer')
    item_column = select_column(ranked_columns, 'item number')

    print(f"Selected date column: {date_column}")
    print(f"Selected manufacturer column: {manufacturer_column}")
    print(f"Selected item column: {item_column}")

    # Convert the date column to a usable datetime format
    data[date_column] = convert_excel_date(data[date_column])

    # Drop rows with NaN dates to avoid errors
    data = data.dropna(subset=[date_column])

    # Extract week number and year from the date column
    data['Week'] = data[date_column].dt.isocalendar().week
    data['Year'] = data[date_column].dt.year

    # Analyze Manufacturers
    manufacturer_count = data.groupby(['Year', 'Week', manufacturer_column]).size().reset_index(name='Frequency')
    avg_invoices_manufacturer = manufacturer_count.groupby(manufacturer_column)['Frequency'].mean().reset_index(name='Average Weekly Invoices')
    manufacturer_total_freq = data[manufacturer_column].value_counts().reset_index()
    manufacturer_total_freq.columns = [manufacturer_column, 'Total Frequency']
    manufacturer_total_freq['Percent Contribution'] = (manufacturer_total_freq['Total Frequency'] / manufacturer_total_freq['Total Frequency'].sum() * 100).round(1)
    manufacturer_total_freq['Cumulative Percentage'] = manufacturer_total_freq['Percent Contribution'].cumsum().round(1)
    top_80_manufacturers = manufacturer_total_freq[manufacturer_total_freq['Cumulative Percentage'] <= 80]
    top_manufacturers = top_80_manufacturers[manufacturer_column].tolist()

    # Analyze Manufacturer-Item Combinations for Top Manufacturers
    top_manufacturer_item_count = data[data[manufacturer_column].isin(top_manufacturers)].groupby(['Year', 'Week', manufacturer_column, item_column]).size().reset_index(name='Frequency')
    avg_invoices_item = top_manufacturer_item_count.groupby([manufacturer_column, item_column])['Frequency'].mean().reset_index(name='Average Weekly Invoices')
    item_total_freq = data[data[manufacturer_column].isin(top_manufacturers)].groupby([manufacturer_column, item_column]).size().reset_index(name='Total Frequency')
    item_total_freq['Percent Contribution'] = (item_total_freq['Total Frequency'] / item_total_freq['Total Frequency'].sum() * 100).round(1)
    item_total_freq['Cumulative Percentage'] = item_total_freq['Percent Contribution'].cumsum().round(1)
    top_80_items = item_total_freq[item_total_freq['Cumulative Percentage'] <= 80][[manufacturer_column, item_column]]

    # Forecast Future Usage for Top 80% Items for Each Top Manufacturer
    forecast_results = forecast_usage(data, date_column, manufacturer_column, item_column, top_80_items.values.tolist())

    # Save the results to a new Excel file with eight sheets
    output_file = 'manufacturer_and_item_analysis.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        manufacturer_count.to_excel(writer, sheet_name='Weekly Manufacturer Frequency', index=False)
        avg_invoices_manufacturer.to_excel(writer, sheet_name='Average Weekly Invoices (Man)', index=False)
        top_80_manufacturers.to_excel(writer, sheet_name='Top 80% Manufacturers', index=False)
        forecast_results.to_excel(writer, sheet_name='Forecast Results (Items)', index=False)
        
        top_manufacturer_item_count.to_excel(writer, sheet_name='Weekly Man-Item Frequency', index=False)
        avg_invoices_item.to_excel(writer, sheet_name='Average Weekly Invoices (Items)', index=False)
        top_80_items.to_excel(writer, sheet_name='Top 80% Man-Items', index=False)

    # Apply formatting to the percentage columns in the 'Top 80% Manufacturers' and 'Top 80% Man-Items' sheets
    from openpyxl import load_workbook

    workbook = load_workbook(output_file)
    worksheet_manufacturers = workbook['Top 80% Manufacturers']
    for col in ['C', 'D']:
        for cell in worksheet_manufacturers[col][1:]:
            cell.number_format = '0.0%'

    worksheet_items = workbook['Top 80% Man-Items']
    for col in ['C', 'D']:
        for cell in worksheet_items[col][1:]:
            cell.number_format = '0.0%'

    workbook.save(output_file)

    print(f"Analysis saved to {output_file}")

if __name__ == "__main__":
    main()

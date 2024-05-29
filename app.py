import os
import pandas as pd
import re
from transformers import pipeline
import streamlit as st
from io import BytesIO

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
    st.write(f"Select a {column_type} column:")
    options = [header for header, score in ranked_columns[column_type]]
    choice = st.selectbox(f"Select a {column_type} column", options)
    return choice

def convert_excel_date(date_series):
    # Handle both datetime and numeric date formats
    if pd.api.types.is_numeric_dtype(date_series):
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(date_series, 'D')
    else:
        return pd.to_datetime(date_series, errors='coerce')

def main():
    st.title("Sales Analysis Tool")

    # File uploader
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xlsb"])

    if uploaded_file is not None:
        # Read the uploaded file
        try:
            if uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file, engine='openpyxl')
            elif uploaded_file.name.endswith('.xlsb'):
                data = pd.read_excel(uploaded_file, engine='pyxlsb')
            
            st.write("File uploaded successfully.")
            headers = data.columns.tolist()
            ranked_columns = recommend_columns(headers)

            date_column = select_column(ranked_columns, 'date')
            manufacturer_column = select_column(ranked_columns, 'manufacturer')
            item_column = select_column(ranked_columns, 'item number')

            st.write(f"Selected date column: {date_column}")
            st.write(f"Selected manufacturer column: {manufacturer_column}")
            st.write(f"Selected item column: {item_column}")

            # Convert the date column to a usable datetime format
            data[date_column] = convert_excel_date(data[date_column])

            # Drop rows with NaN dates to avoid errors
            data = data.dropna(subset=[date_column])

            # Extract week number and year from the date column
            data['Week'] = data[date_column].dt.isocalendar().week
            data['Year'] = data[date_column].dt.year

            # Analyze Manufacturers
            st.write("Analyzing manufacturers...")
            manufacturer_count = data.groupby(['Year', 'Week', manufacturer_column]).size().reset_index(name='Frequency')
            avg_invoices_manufacturer = manufacturer_count.groupby(manufacturer_column)['Frequency'].mean().reset_index(name='Average Weekly Invoices')
            manufacturer_total_freq = data[manufacturer_column].value_counts().reset_index()
            manufacturer_total_freq.columns = [manufacturer_column, 'Total Frequency']
            manufacturer_total_freq['Percent Contribution'] = (manufacturer_total_freq['Total Frequency'] / manufacturer_total_freq['Total Frequency'].sum() * 100).round(1)
            manufacturer_total_freq['Cumulative Percentage'] = manufacturer_total_freq['Percent Contribution'].cumsum().round(1)
            top_80_manufacturers = manufacturer_total_freq[manufacturer_total_freq['Cumulative Percentage'] <= 80]
            top_manufacturers = top_80_manufacturers[manufacturer_column].tolist()

            # Analyze Manufacturer-Item Combinations for Top Manufacturers
            st.write("Analyzing manufacturer-item combinations...")
            top_manufacturer_item_count = data[data[manufacturer_column].isin(top_manufacturers)].groupby(['Year', 'Week', manufacturer_column, item_column]).size().reset_index(name='Frequency')
            avg_invoices_item = top_manufacturer_item_count.groupby([manufacturer_column, item_column])['Frequency'].mean().reset_index(name='Average Weekly Invoices')
            item_total_freq = data[data[manufacturer_column].isin(top_manufacturers)].groupby([manufacturer_column, item_column]).size().reset_index(name='Total Frequency')
            item_total_freq['Percent Contribution'] = (item_total_freq['Total Frequency'] / item_total_freq['Total Frequency'].sum() * 100).round(1)
            item_total_freq['Cumulative Percentage'] = item_total_freq['Percent Contribution'].cumsum().round(1)
            top_80_items = item_total_freq[item_total_freq['Cumulative Percentage'] <= 80][[manufacturer_column, item_column]]

            # Save the results to a new Excel file with four sheets
            st.write("Saving results to Excel file...")
            output_file = BytesIO()
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                manufacturer_count.to_excel(writer, sheet_name='Weekly Manufacturer Frequency', index=False)
                avg_invoices_manufacturer.to_excel(writer, sheet_name='Average Weekly Invoices (Man)', index=False)
                top_80_manufacturers.to_excel(writer, sheet_name='Top 80% Manufacturers', index=False)
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

            st.write("Analysis complete. You can download the result below.")
            st.download_button(
                label="Download Excel file",
                data=output_file.getvalue(),
                file_name="manufacturer_and_item_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

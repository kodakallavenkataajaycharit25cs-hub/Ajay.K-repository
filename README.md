Regex file explaination
Baby Names Popularity Analysis
This project analyzes baby names popularity from a provided HTML file (baby1990.html). It extracts the year of the data, parses the top male and female baby names along with their ranks, and visualizes the top male names using a bar chart.

Features
Extracts the year from the HTML file.

Extracts and displays the top 10 male and female baby names with ranks.

Visualizes the top 10 male baby names by their ranks using a bar chart.

Creates a dictionary of names with their best ranks for both male and female names, displaying the top entries.

Usage
Place the HTML file (e.g., baby1990.html) in the project directory.

Run the Python script to:

Extract the year of the data.

Extract and display the top names and their ranks.

Generate a bar chart for the top 10 male names.

Display a dictionary of names with their ranks.

Dependencies
matplotlib

pandas

re (regular expressions, built-in Python library)

Install dependencies via pip if needed:

bash
pip install matplotlib pandas
Code Overview
extract_year(filename): Reads the HTML file and extracts the year of birth popularity.

extract_names_and_ranks(filename): Extracts the top 10 male and female names with ranks into a pandas DataFrame.

Visualization: Displays a bar chart of the top 10 male names by rank.

extract_names_to_dict(filename): Creates and displays a dictionary mapping names to their best rank across genders.

Example Output
Prints the extracted year (e.g., 1990).

Shows a table of the top 10 male and female names with ranks.

Displays a bar chart for the top male names.

Prints a table of names with their best ranks

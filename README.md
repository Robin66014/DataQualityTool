# DQ Analyzer toolkit
An automated tool for the quantitative assessment of data quality
Python version 3.10

The tool is still under construction, and has as final goal to asses data quality and find the most common data issues in your data set. 
Accepted file types: .csv, .arff, .xls, .xlsx, .parquet

Working of the tool (in web app):
1. Upload your data file 
2. Adjust (if necessary) your column data types that were automatically inferred & in the drop-down menu select your target column
3. Press 'Run checks', your data quality report will now be created
4. Press 'Run additional checks', now potential label errors will be searched for and a baseline performance assessment is given. Optional: perform a bias analysis yourself\
5. Accept & apply the recommended data quality remediations with just a few clicks to improve your dataset!


# How to get the tool running:
1. Pull this github or download the .zip folder
2. install the requirements by navigating to the correct directory using cd, next type in command prompt: pip install -r requirements.txt
3. Make sure you have the correct versions of the packages installed, newer deepchecks versions will bug the tool (--> pip show deepchecks)
3. run the app using the following command: py dashapp.py
4. your app will now be run on http://127.0.0.1:8050/ (in your webbrowser)



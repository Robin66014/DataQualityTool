# DataQualityTool
An automated tool for the quantitative measurement of data quality

The tool is still under construction, and has as final goal to asses data quality and find the most common data issues in your data set. 
The three data types accepted by the tool: Structured data, image data (.ARFF files and folders with JPGs), and time-series data.


Working of the tool (in web app):
1. Upload your data file and select in the drop-down menu your data type.
2. Press 'next page', your feature types will be automatically inferred. If necessary, change the feature types in the columns
3. In the drop-down menu, select your target column, if there is none select 'N/A'
4. Press 'Run checks', your checks will now automatically be runm NOTE: THIS MAY TAKE A WHILE
5. The page you will see now will contain a complete overview of the potential issues in your dataset, and actionable advise on how to improve it.
6. Press 'Advanced Settings' if you want to remove some checks that are irrelevant for your specific case.
7. Press re-run checks to go back to the overview page, this time without the irrelevant checks.


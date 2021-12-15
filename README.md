# ITC6001
ITC6001 Introduction to Big Data

The purpose of this project is to showcase the differences in speed, size and accuracy between traditional batch processing and stream processing methodologies.  
The hypothesis made at the begining of writing is that by using stream processing techniques and not storing any of the actual data, we can maintain the accuracy of the results while dramatically reducing the workload and total memory required for analyzing and storing the data. 
This will done by finding the heavy hitters (most frequent appearances) for two parameters in a dataset
The batch processing part of this project will be done by utilizing dictionaries and Pandas Dataframes to store and analyze data, while the stream processing will take place using the Count Mins Sketch (CMS) and HyperLogLog (HLL) algorithms. 
The goal is to compare the frequency counts (Heavy Hitters) for two parameters in a given dataset using both processing methods. 
All results will be aggregated and summarized using tables and graphs as outputs. 
The code that will process the data is written in Python 3.9.5 and the main processing libraries used are Pandas, Numpy, Pyprobables, Datasketch and Matplotlib.

****************
Run Instructions
****************
Place *extracted* dataset in the same directory as .py file. 
The five JSON files need to have the the tweets.json.x format
The code will create two directories
First is the csv_files -> will include all .csv exports for every 10% of dataset
Second is graphs -> will include runtime graphs that are evaluated on the report
The code will also produce Runtime_Summary.log file at the .py directory that includes data about every program itteration
The same logfile is saved in its final form as a .csv file in the same directory
To run the program again remove all files created by the program
****************
Code written and executed using Python 3.9.5 and Visual Studio Code
Created entirely by: Ilias Siafakas

"""
 ,---.        ,--.,--.,--.                                                                                                                                          
'   .-'       `--'|  |`--' ,--,--. ,---.                                                                                                                            
`.  `-.       ,--.|  |,--.' ,-.  |(  .-'                                                                                                                            
.-'    |,----.|  ||  ||  |\ '-'  |.-'  `)                                                                                                                           
`-----' '----'`--'`--'`--' `--`--'`----'                                                                                                                            
,--.,--------. ,-----.  ,--.  ,--.    ,--.   ,--.                                                                                                                   
|  |'--.  .--''  .--./ /  .' /    \  /    \ /   |                                                                                                                   
|  |   |  |   |  |    |  .-.|  ()  ||  ()  |`|  |                                                                                                                   
|  |   |  |   '  '--'\\   o |\    /  \    /  |  |                                                                                                                   
`--'   `--'    `-----' `---'  `--'    `--'   `--'                                                                                                                   
,--.          ,--.                   ,--.                ,--.  ,--.                     ,--.             ,-----.  ,--.           ,------.            ,--.           
|  |,--,--, ,-'  '-.,--.--. ,---.  ,-|  |,--.,--. ,---.,-'  '-.`--' ,---. ,--,--,     ,-'  '-. ,---.     |  |) /_ `--' ,---.     |  .-.  \  ,--,--.,-'  '-. ,--,--. 
|  ||      \'-.  .-'|  .--'| .-. |' .-. ||  ||  || .--''-.  .-',--.| .-. ||      \    '-.  .-'| .-. |    |  .-.  \,--.| .-. |    |  |  \  :' ,-.  |'-.  .-'' ,-.  | 
|  ||  ||  |  |  |  |  |   ' '-' '\ `-' |'  ''  '\ `--.  |  |  |  |' '-' '|  ||  |      |  |  ' '-' '    |  '--' /|  |' '-' '    |  '--'  /\ '-'  |  |  |  \ '-'  | 
`--'`--''--'  `--'  `--'    `---'  `---'  `----'  `---'  `--'  `--' `---' `--''--'      `--'   `---'     `------' `--'.`-  /     `-------'  `--`--'  `--'   `--`--'
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


"""
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
"""


import json
import pandas as pd
import time
from pandas.core.indexes.base import Index
from probables import CountMinSketch
import re
import numpy as np
from datasketch import HyperLogLog
import sys
import os
import matplotlib.pyplot as plt
import math
import logging


def create_answer_dir(name): #creates directory for deliverables
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, rf'{name}')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)


def find_hashtags(text):
    regex = "#(\w+)" # the regular expression
    return re.findall(regex, text) # extracting the hashtags


def populateTags(dict, cms, hll, tweet,t): #take the tweet, retrieve all hashtags, pass them through CMS and HLL, add to dict.
    tags = find_hashtags(tweet['text'])
    for tag in tags:
        start_time = time.time()
        cms = countmin(cms,tag)
        hll.update(str(tag).encode('utf8'))
        t += time.time() - start_time
        if tag not in dict:
            dict[tag] = 1
        else:
            dict[tag] += 1
    return dict,cms,hll,t


def populateUsers(dict, cms, hll, tweet, t): #take the tweet, find the username, pass it through CMS and HLL, add to dict.
    start_time = time.time()
    name = tweet['user']['screen_name']
    cms = countmin(cms,name)
    hll.update(str(name).encode('utf8'))
    t += time.time() - start_time
    if name not in dict:
        dict[name] = 1
    else:
        dict[name] += 1
    return dict, cms, hll, t


def countmin(cms,value): #simple add to given sketch
    cms.add(value)
    return cms


def setup_df(dict,cols): 
    df = pd.DataFrame(dict.items(), columns =cols) #convert dictionary to dataframe
    df = df.sort_values('Frequency',ascending=False) #sort and reset indexes -- this will create a dataframe with index 1 being highest frequency value
    df = df.reset_index(drop=True)
    df["CMS"] = np.nan #add required columns for future calculations and metrics
    df["Difference"] = np.nan
    df["Rank Difference"] = np.nan
    return df


def create_columns(df, cms,col): 
    for index, row in df.iterrows():
        df.loc[index,'CMS'] = cms.check(row[col])
        df.loc[index,'Difference'] = df.loc[index,'CMS'] - df.loc[index,'Frequency']
    df["Rank Difference"] = df["Frequency"].rank(method='first',ascending=False) - df["CMS"].rank(method='first',ascending=False)
    return df



def create_frame(dict, cms, cols):
    df = setup_df(dict,cols) #setups Dataframe, sorts based on observed Frequency and adds columns for metrics
    df = create_columns(df,cms,cols[0]) #populate CMS row, find the difference with the actual frequency. col[0] gives the name of the dataframe we created(can be 'Tag' or 'Username')
    #df = df.sort_values('CMS', ascending=False) 
    return df


def main():
    #initialize all objects
    stream_time = 0
    run_time = time.time()
    user_dict = {}
    tag_dict = {}
    user_cms = CountMinSketch(width=2719, depth=7)
    tag_cms = CountMinSketch(width=2719, depth=7)
    user_hll = HyperLogLog()
    tag_hll = HyperLogLog()
    metrics_cols=["Percent of DataSet", "Time of Batch Process", "Time of Stream Process", "Time to Process (s)", "Cummulative Time (s)", "Count of Users", "HLL Count of Users", "MAE Users","MAPE Users (20)", "Count of tags", "HLL Count of Tags", "MAE Tags","MAPE Tags (20)", "Size of Frequency Count Solution (Bytes)", "Size of CMS+HLL (Bytes)"]
    df_summary = pd.DataFrame([],columns=metrics_cols)
    itteration = 0
    graph_x = []
    #create directories and logfile
    create_answer_dir("csv_files") 
    create_answer_dir("graphs")
    logging.basicConfig(filename='Runtime_Summary.log', encoding='utf-8',level=logging.INFO)
    for j in range(5): #for every file (fixed number)
        print(f"start of file {j}")
        with open(f'tweets.json.{j}','r', encoding='UTF-8') as f:
            i = 0
            filelength = 0 #reset line counter
            filelength = len(f.readlines()) #get the number of rows in the file
            f.seek(0) #set cursor back to first line
            for row in f:
                tweet = json.loads(row) #convert string to json object
                tag_dict,tag_cms,tag_hll,stream_time = populateTags(tag_dict,tag_cms, tag_hll, tweet,stream_time) #return all the objects filled with the parameter
                user_dict,user_cms,user_hll,stream_time = populateUsers(user_dict,user_cms,user_hll,tweet,stream_time)
                i = i+1
                if i % math.floor((filelength/2)) == 0: #used to calculate the half and end point of each file. math.floor() is used for odd row files. 
                    #the last row is excluded from the exported file, but included in the next pass
                    itteration = int(2*j+np.round(i/(filelength/2),0)) #this could have been a simple incrementer but i prefered a more complicated way of calculating current itteration. basically counts how many itterations the code has gone through
                    df1 = create_frame(user_dict,user_cms,['Username', 'Frequency'])#transfer filled dictionaries to dataframes and perform all necessary calculations. these are recreated every itteration
                    df2 = create_frame(tag_dict,tag_cms,['Tag', 'Frequency'])

                    df1.head(20).to_csv(f"csv_files\HH_Usernames_{itteration*10}%.csv") #produce csv files with 10% snapshot
                    df2.head(20).to_csv(f"csv_files\HH_Tags_{itteration*10}%.csv")
                    ######create summary
                    summary_row = {
                        "Percent of DataSet": itteration*10,
                        "Time of Stream Process": stream_time,
                        "Time of Batch Process": time.time()-run_time-stream_time, #batch time is total execution time - anything that was processed for the stream method
                        "Time to Process (s)": time.time()-run_time,
                        "Cummulative Time (s)": 0, # for now is zero, will be updated after append using the cumsum() method
                        "Count of Users": len(df1.index),
                        "HLL Count of Users": user_hll.count(),
                        "MAE Users": df1['Difference'].sum()/len(df1.index),
                        "MAPE Users (20)": np.mean(np.abs((np.array(df1['Frequency'].head(20)) - np.array(df1['CMS'].head(20))) / np.array(df1['Frequency'].head(20)))), #took only the first 20 values because after the CMS predictions alter the results
                        "Count of tags": len(df2.index),
                        "HLL Count of Tags": tag_hll.count(),
                        "MAE Tags": df2['Difference'].sum()/len(df2.index),
                        "MAPE Tags (20)": np.mean(np.abs((np.array(df2['Frequency'].head(20)) - np.array(df2['CMS'].head(20))) / np.array(df2['Frequency'].head(20)))),
                        "Size of Frequency Count Solution (Bytes)": sys.getsizeof(df1)+sys.getsizeof(df2)+sys.getsizeof(user_dict)+sys.getsizeof(tag_dict), #counts approximate size of both dictionaries and Dataframes
                        "Size of CMS+HLL (Bytes)": sys.getsizeof(user_hll)+sys.getsizeof(tag_hll)+sys.getsizeof(user_cms)+sys.getsizeof(tag_cms) #counts size of HLL and CMS for both usernames and hashtags -- this should remain fixed size
                        }
                    df_summary = df_summary.append(summary_row, ignore_index=True)
                    df_summary["Cummulative Time (s)"] = df_summary["Time to Process (s)"].cumsum() #create cumulative time
                    
                    logging.info(df_summary)
                    
                    graph_x.append(itteration*10/100) 
                    print(f"{itteration*10}% done")
                    stream_time = 0 
                    run_time = time.time()
            print(f"end of file {j}")
    df_summary.to_csv('Runtime_Summary.csv') #produce final summary csv


    ####
    #Figure 1 Stream vs Batch execution time
    ####
    plt.xlabel('Dataset %')
    plt.ylabel('Time (s)')
    color = 'tab:red'    
    plt.figure(1)
    plt.plot(graph_x, df_summary['Time of Batch Process'], label = 'Batch', color=color)
    color = 'tab:blue'
    plt.plot(graph_x, df_summary['Time of Stream Process'], label='Stream', color=color)
    plt.legend()
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("graphs\execution_time.png",dpi='figure',format='png')
    

    ####
    #Figure 2 Batch vs Stream Memory size
    ####
    plt.figure(2)
    plt.xlabel('Dataset %')
    # Set the y axis label of the current axis.
    plt.ylabel('Memory Size (B)')
    color = 'tab:red'    
    plt.plot(graph_x, df_summary['Size of Frequency Count Solution (Bytes)'], label = 'Batch', color=color)
    color = 'tab:blue'
    plt.plot(graph_x, df_summary['Size of CMS+HLL (Bytes)'], label='Stream', color=color)
    plt.legend()
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("graphs\memory_size.png",dpi='figure',format='png')


    ####
    #Figure 3 frequency vs HLL
    ####
    plt.figure(3)
    plt.xlabel('Dataset %')
    plt.ylabel('Count')
    color = 'tab:red'    
    plt.plot(graph_x, df_summary["Count of Users"], label='Count of Users', color=color)
    color = 'tab:blue'
    plt.plot(graph_x, df_summary["HLL Count of Users"], label='HLL Count of Users', color=color)
    color = 'tab:green'
    plt.plot(graph_x, df_summary['Count of tags'], label='Count of Tags', color=color)
    color = 'tab:orange'
    plt.plot(graph_x, df_summary['HLL Count of Tags'], label='HLL Count of Tags', color=color)
    plt.legend()
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("graphs\Frequency_vs_HLL.png",dpi='figure',format='png')


    ####
    #Figure 3 process time vs total frequency
    ####
    plt.figure(4)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Dataset %')
    ax1.set_ylabel('time (s)', color=color)
    ax1.plot(graph_x, df_summary["Cummulative Time (s)"], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # create a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Total Frequency', color=color)  # we already handled the x-label with ax1
    ax2.plot(graph_x, df_summary['Count of Users']+df_summary['Count of tags'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("graphs\\time_over_Frequency.png",dpi='figure',format='png')




if __name__ == '__main__':
    main()





"""
 ,---.        ,--.,--.,--.                                                                                                                                          
'   .-'       `--'|  |`--' ,--,--. ,---.                                                                                                                            
`.  `-.       ,--.|  |,--.' ,-.  |(  .-'                                                                                                                            
.-'    |,----.|  ||  ||  |\ '-'  |.-'  `)                                                                                                                           
`-----' '----'`--'`--'`--' `--`--'`----'                                                                                                                            
,--.,--------. ,-----.  ,--.  ,--.    ,--.   ,--.                                                                                                                   
|  |'--.  .--''  .--./ /  .' /    \  /    \ /   |                                                                                                                   
|  |   |  |   |  |    |  .-.|  ()  ||  ()  |`|  |                                                                                                                   
|  |   |  |   '  '--'\\   o |\    /  \    /  |  |                                                                                                                   
`--'   `--'    `-----' `---'  `--'    `--'   `--'                                                                                                                   
,--.          ,--.                   ,--.                ,--.  ,--.                     ,--.             ,-----.  ,--.           ,------.            ,--.           
|  |,--,--, ,-'  '-.,--.--. ,---.  ,-|  |,--.,--. ,---.,-'  '-.`--' ,---. ,--,--,     ,-'  '-. ,---.     |  |) /_ `--' ,---.     |  .-.  \  ,--,--.,-'  '-. ,--,--. 
|  ||      \'-.  .-'|  .--'| .-. |' .-. ||  ||  || .--''-.  .-',--.| .-. ||      \    '-.  .-'| .-. |    |  .-.  \,--.| .-. |    |  |  \  :' ,-.  |'-.  .-'' ,-.  | 
|  ||  ||  |  |  |  |  |   ' '-' '\ `-' |'  ''  '\ `--.  |  |  |  |' '-' '|  ||  |      |  |  ' '-' '    |  '--' /|  |' '-' '    |  '--'  /\ '-'  |  |  |  \ '-'  | 
`--'`--''--'  `--'  `--'    `---'  `---'  `----'  `---'  `--'  `--' `---' `--''--'      `--'   `---'     `------' `--'.`-  /     `-------'  `--`--'  `--'   `--`--'
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""




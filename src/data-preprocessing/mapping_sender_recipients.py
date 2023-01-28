#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import csv

def create_sender_recipient_list(df):

    
    #to create dataframe that has From,to column
    from_list = []
    to_list = []
    for index, series in df.iterrows():
        for value in df.iloc[index]['Recipient']:
            if value != '':
                to_list.append(value)
                from_list.append(df.iloc[index]['Sender'])

        for value in df.iloc[index]['CC']:
            if value != '':
                to_list.append(value)
                from_list.append(df.iloc[index]['Sender'])

        for value in df.iloc[index]['BCC']:
            if value != '':
                to_list.append(value)
                from_list.append(df.iloc[index]['Sender'])

    network_df = pd.DataFrame(list(zip(from_list, to_list)), columns = ['From', 'To'])
    network_df.drop_duplicates(inplace = True)
    networkDf = network_df.loc[network_df.From.str.contains(r'.*@enron.com'), :]
    create_unique_employee_list(networkDf)

def create_unique_employee_list(network_df):

    #This function is to create list of
    #unique employees and assign it to unique id
    #create list of all the uniquq employee.
    
    sender_list = set()
    for i in range(len(network_df)):
        sender_list.add(network_df.iloc[i]['From'])
        sender_list.add(network_df.iloc[i]['To'])

    index = []
    for i in range(len(sender_list)):
        index.append(i)

    mapping_df = pd.DataFrame(list(zip(index, sender_list)), columns = ['ID', 'Nodes'])
    mapping_df.drop_duplicates(inplace = True)
    mapping_df.to_pickle('mapping_id_node')

    map_employee_to_id(network_df, mapping_df)

def map_employee_to_id(network_df, mapping_df):

    network_df = network_df.merge(mapping_df,left_on = 'From', right_on = 'Nodes')
    network_df.drop(columns = ['Nodes'], inplace = True)
    network_df.rename(columns = {'ID' : 'From_ID'}, inplace = True)
    network_df = network_df.merge(mapping_df, left_on = 'To', right_on = 'Nodes').drop(columns = ['Nodes'])
    network_df.rename(columns = {'ID' : 'To_ID'}, inplace = True)

    #save the dataframe
    network_df.to_pickle('mappingid_from_to')

if __name__ == '__main__':

    #ToDo: read the feature_df once it is created   
    df = pd.read_pickle('feature_df')

    create_sender_recipient_list(df)
import pandas as pd
from data_conversion import create_sender_receiver_static_df, create_sender_receiver_dynamic_df, create_sender_receiver_dynamic_df_email
import re
import csv
class FilterDataset():

    def count_max_sent_email(df):

        count_df = df.value_counts(subset = ['Sender'])
        count_df.to_csv('sender_count_full.csv')

    def filter_df(df, weight_arg, graph_type):
        top_list = []
        email_regex = re.compile(r'.*@enron.com')

        # filter the dataframe based on only enron employess
        df = df.loc[df['Sender'].str.contains(email_regex)]

        with open('sender_count_full.csv', 'r') as file:
            sender_info = csv.DictReader(file)
            for row in sender_info:
                if int(row['count']) > 500:
                #if row['Sender'].contains(email_regex):
                    top_list.append(row['Sender'])
                else:
                    continue
    
        # filter dataframe further and select only people who are top 10
        # sender's and the recipients of those sender's
        '''top_list = ['kay.mann@enron.com', 'jeff.dasovich@enron.com', 'tana.jones@enron.com', 
                    'sara.shackleton@enron.com', 'vince.kaminski@enron.com', 'chris.germany@enron.com']'''
        filtered_df = df.loc[df['Sender'].isin(top_list)]
        # creating dataframe for sender and recipient information
        # filter only enron employeess in the 'To' field
        if graph_type == "static":
            sender_receiver_df = create_sender_receiver_static_df(filtered_df)
        elif graph_type == "dynamic":
            sender_receiver_df = create_sender_receiver_dynamic_df(filtered_df)
    
        '''if weight_arg == 'weighted' and graph_type == "dynamic":
            #sender_receiver_df = sender_receiver_df.groupby(['From', 'To', 'TimeSent']).size().reset_index(name = 'counts')
            sender_receiver_df = sender_receiver_df.loc[sender_receiver_df['To'].str.contains(email_regex)]'''

        if weight_arg == "weighted" and graph_type == "static":
            sender_receiver_df = sender_receiver_df.groupby(['From', 'To']).size().reset_index(name = 'counts')
            sender_receiver_df = sender_receiver_df.loc[sender_receiver_df['To'].str.contains(email_regex)]

        else:
            #sender_receiver_df.drop_duplicates(subset = ['From', 'To'], inplace = True)
            sender_receiver_df = sender_receiver_df.loc[sender_receiver_df['To'].str.contains(email_regex)]

        return top_list, sender_receiver_df
    
    def create_sender_receiver_list_email(df, graph_type):
        if graph_type == "static":
            sender_receiver_df = create_sender_receiver_static_df(df)
        else:
            sender_receiver_df = create_sender_receiver_dynamic_df_email(df)

        return sender_receiver_df
import pandas as pd
import time
from dateutil import parser
from datetime import datetime

# This function is to create dataframe for sender and reciever 
def create_sender_receiver_static_df(df):
    from_list = []
    to_list = []
    
    for row in df.itertuples():
        sender = row.Sender
        for recipient in row.Recipient:
            from_list.append(sender)
            to_list.append(recipient)
        for cc in row.CC:
            from_list.append(sender)
            to_list.append(cc)
        for bcc in row.BCC:
            from_list.append(sender)
            to_list.append(bcc)
    sender_receiver_df = pd.DataFrame(list(zip(from_list, to_list)), columns = ['From', 'To'])
    return sender_receiver_df

# adding time information for dynamic setting
def create_sender_receiver_dynamic_df(df):
    from_list = []
    to_list = []
    date_list = []

    for row in df.itertuples():
    
        sender = row.Sender
        date = row.date 
        #print(date)
        date = time.mktime(parser.parse(row.date).timetuple())
        date = datetime.fromtimestamp(date).strftime("%Y-%m-%d")
        #print(date)
        #print("After converting date", date)
        for recipient in row.Recipient:
            from_list.append(sender)
            to_list.append(recipient)
            date_list.append(date)
        for cc in row.CC:
            from_list.append(sender)
            to_list.append(cc)
            date_list.append(date)
        for bcc in row.BCC:
            from_list.append(sender)
            to_list.append(bcc)
            date_list.append(date)
        
    sender_receiver_df = pd.DataFrame(list(zip(from_list, to_list, date_list)), columns = ['From', 'To', 'TimeSent'])
    return sender_receiver_df

# adding time information for dynamic setting
def create_sender_receiver_dynamic_df_email(df):
    from_list = []
    to_list = []
    date_list = []

    for row in df.itertuples():
        try:
            sender = row.Sender
            date = row.date 
            #print(date)
            date = time.mktime(parser.parse(row.date).timetuple())
            date = datetime.fromtimestamp(date).strftime("%Y-%m-%d")
            #print(date)
            #print("After converting date", date)
            for recipient in row.Recipient:
                from_list.append(sender)
                to_list.append(recipient)
                date_list.append(date)
            for cc in row.CC:
                from_list.append(sender)
                to_list.append(cc)
                date_list.append(date)
            for bcc in row.BCC:
                from_list.append(sender)
                to_list.append(bcc)
                date_list.append(date)
        except:
            #print("----")
            continue
            
    sender_receiver_df = pd.DataFrame(list(zip(from_list, to_list, date_list)), columns = ['From', 'To', 'TimeSent'])
    return sender_receiver_df
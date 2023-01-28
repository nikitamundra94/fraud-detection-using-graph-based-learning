#!/usr/bin/env python
# coding: utf-8


import os
import datetime
from email.parser import Parser
import re
import csv
import pandas as pd
from glob import glob
import time
from dateutil import parser
import numpy as np
import itertools
from email.parser import HeaderParser
import spacy

nlp = spacy.load("en_core_web_sm")

#specify the path to the enron datase
dir = r'/p/fm/MundraNikita/Project/maildir/*/*'

def extract_mail_addresses(header_value:str):
        result = set()
        doc = nlp(header_value)
        for token in doc:
            if token.like_email:
                result.add(token.text)
        return result

def feature_extrat():
    to_list = []
    subject_list = []
    cc_list = []
    bcc_list = []
    email_body = []
    x_origin = []
    mail_date = []
    mime = []
    from_list = []
    content_type = []
    content_encoding = []
    to_uname = []
    i = 0
    dict_dataframe = {}
    for path in glob(f'{dir}/*',recursive=True):
        print(path)
        try:
            with open(path, 'r') as f:
                data = f.read()
                email_message = HeaderParser().parsestr(data.strip())
                email_dict = dict(email_message)

                if 'Message-ID' not in email_dict:
                    continue
                mail_date.append(email_dict["Date"] if "Date" in email_dict else "")
                subject_list.append(email_dict["Subject"] if "Subject" in email_dict else "")
                x_origin.append(email_dict["X-Origin"] if "X-Origin" in email_dict else "")
                mime.append(email_dict["Mime-Version"] if "Mime-Version" in email_dict else "")
                email_body.append(email_message.get_payload())
                from_list.append(email_dict['From'] if 'From' in email_dict else "")
                content_type.append(email_dict['Content-Type'] if "Content-Type" in email_dict else "")
                content_encoding.append(email_dict['Content-Transfer-Encoding'] if "Content-Transfer-Encoding" in email_dict else "")

                to_list.append(list(extract_mail_addresses(email_dict['To'])) if 'To' in email_dict else '' )
                cc_list.append(list(extract_mail_addresses(email_dict['Cc'])) if 'Cc' in email_dict else '' )
                bcc_list.append(list(extract_mail_addresses(email_dict['Bcc'])) if 'Bcc' in email_dict else '')
        except:
            
            continue

    
  
    dict_dataframe['Sender'] = from_list
    dict_dataframe['Recipient'] = to_list
    dict_dataframe['CC'] = cc_list
    dict_dataframe['BCC'] = bcc_list
    dict_dataframe['date'] = mail_date
    dict_dataframe['subject'] = subject_list
    dict_dataframe['Text'] = email_body
    dict_dataframe['Mime'] = mime
    dict_dataframe['OriginFolder'] = x_origin
    dict_dataframe['ContentType'] = content_type
    dict_dataframe['ContentEncoding'] = content_encoding

    df = pd.DataFrame(dict_dataframe)

    #save the dataframe 
    df.to_pickle('thesis_full_enron')

if __name__ == '__main__':
    feature_extrat()
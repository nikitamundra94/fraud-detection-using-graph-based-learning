import pandas as pd
import re

def get_feat_matrix(df, top_list):

    feature_df = pd.DataFrame()
    
    feature_df['Sender'] = df['Sender']

    feature_df['OnlyToEnron'] = df['Recipient'].apply(calculate_only_enron_employee)
    feature_df['TotalRecipient'] = df['Recipient'].apply(lambda x : len(x)) 
    feature_df['NotToEnron'] = df['Recipient'].apply(calculate_not_enron_employee)

    feature_df['OnlyCCEnron'] = df['CC'].apply(calculate_only_enron_employee)
    feature_df['TotalCC'] = df['CC'].apply(lambda x : len(x))
    feature_df['NotCCEnron'] = df['CC'].apply(calculate_not_enron_employee)

    feature_df['OnlyBCCEnron'] = df['BCC'].apply(calculate_only_enron_employee)
    feature_df['TotalBCC'] = df['BCC'].apply(lambda x : len(x))
    feature_df['NotBCCEnron'] = df['BCC'].apply(calculate_not_enron_employee)

    #calculate average subjectlength of the email
    feature_df['SubjectLength'] = df['subject'].apply(calculate_subject_length)

    #calculate character count in the email
    #feature_df['DifferentSymbolsSubject'] = df['subject'].apply(get_character_count)

    #calculate number of words in the email body
    feature_df['ContentWordCount'] = df['Text'].apply(lambda x : len(x.split()))
    
    feature_df['Mime'] = df['Mime']
    feature_df['ContentType'] = df['ContentType']
    feature_df['ContentEncoding'] = df['ContentEncoding']

    #calculate average for the above features
    average_feature_df = feature_df.groupby(['Sender']).apply(average_feature_summary)

    #round to 2 decimal places
    #feature_matrix = average_feature_df

    # remove the emails saved in draft 
    # checking condition if atleast field is not equal to zero
    # append that row to f1_matrix
    f1_matrix = average_feature_df[average_feature_df[['AverageNumberTo','AverageNumberCc', 'AverageNumberBcc']].ne(0).any(1)]
    

    #create list of all the employees not in sender list
    #not_in_sender_list = get_not_in_sender_list(sender_receiver_df)
    
    # create list of all the employees not in sender list
    not_sender = get_not_sender(f1_matrix, top_list)

    #insert zeroes entries for employees not in f1_matrix
    full_feature_matrix = get_full_feature_matrix(f1_matrix, not_sender)
    feature_matrix = full_feature_matrix.round(3)
    #feature_matrix.to_csv('normalize.csv')
    
    # create node order list to pass it to adjacency matrix
    # in order to maintain the same node order for both 
    # feature and adjacency matrix
    node_order_list = list(feature_matrix.index)
    
    #full_feature_matrix.to_csv('full_matrix.csv')
    return feature_matrix, node_order_list

def calculate_only_enron_employee(recipient_list:list):
    email_regex = re.compile(r'.*@enron.com')
    count = 0
    for i in recipient_list:
        if email_regex.search(i):
            count = count+1

    if count>0:
        return (count/len(recipient_list))  
    else:
        return 0

def calculate_not_enron_employee(recipient_list:list):
    email_regex = re.compile(r'.*@enron.com')
    count = 0
    for i in recipient_list:
        if not email_regex.search(i):
            count = count+1

    if count>0:
        return (count/len(recipient_list))  
    else:
        return 0

def calculate_subject_length(x):
        try:
            subject_len = len(x)
        except:
            subject_len = 0
        return subject_len

def get_character_count(x):
    try:
       character_count = len(re.sub('[A-Za-z0-9\s]+', '', x))
    except:
        character_count = 0

    return character_count

def average_feature_summary(x):
    result = {
        'EnronMailsTo' : x['OnlyToEnron'].mean(),
        'AverageNumberTo' : x['TotalRecipient'].mean(),
        'OtherMailsTo' :x['NotToEnron'].mean(),
        'EnronMailsCc' :x['OnlyCCEnron'].mean(),
        'OtherMailsCc' :x['NotCCEnron'].mean(),
        'AverageNumberCc': x['TotalCC'].mean(),
        'EnronMailsBcc' :x['OnlyBCCEnron'].mean(),
        'OtherMailsBcc' :x['NotBCCEnron'].mean(),
        'AverageNumberBcc' :x['TotalBCC'].mean(),
        'AverageSubjectLength' :x['SubjectLength'].mean(),
        #'AverageDifferentSymbolsSubject' :x['DifferentSymbolsSubject'].mean(),
        'AverageContentLength':x['ContentWordCount'].mean(),
        'MimeVersionsCount' : x['Mime'].nunique(),
        'DifferentCosCount' : x['ContentType'].nunique(),
        'DifferentEncodingsCount' : x['ContentEncoding'].nunique(),
        }
    return pd.Series(result)

'''def get_not_in_sender_list(sender_receiver_df:pd.DataFrame):
    sender_list = sender_receiver_df['From'].unique().tolist()
    receiver_list = sender_receiver_df['To'].unique().tolist()
    
    difference_set = set(receiver_list) - set(sender_list)
    not_in_sender_list = list(difference_set)
    #employee_list = sender_list + not_in_sender_list
    return not_in_sender_list'''

def get_not_sender(f1_matrix, top_list):
    sender_list = f1_matrix.index
    not_sender_list = set(top_list) - set(sender_list)

    not_sender = list(not_sender_list)

    return not_sender
def get_full_feature_matrix(f1_matrix, not_in_sender_list):

    df = pd.DataFrame([[0]*f1_matrix.shape[1]], columns=f1_matrix.columns, index = list(not_in_sender_list)) 
    df.reset_index(inplace = True)
    df.rename(columns = {'index' : 'Sender'}, inplace = True)
    df.set_index('Sender', inplace = True)
    final_featurematrix = pd.concat([f1_matrix,df])
    return final_featurematrix
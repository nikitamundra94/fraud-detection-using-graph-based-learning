import sys
import pandas as pd
from datetime import datetime
from graph import dataframe_to_graph_network, elliptic_dataframe_to_graph_network
from create_adjacency_matrix import adjacency_dynamic_graph
from filterdataset import FilterDataset
from dynnode2vec import dynnode2vec
import numpy as np

#weight_arg = sys.argv[1]
def evolving_enron():

    # reading the dataset as pkl file
    df = pd.read_pickle('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/thesis_full_enron')
    #FilterDataset.count_max_sent_email(df)
    top_list, sender_receiver_df = FilterDataset.filter_df(df, weight_arg, "dynamic")

    dynamic_sender_reciever_df = sender_receiver_df.sort_values(by = ['TimeSent'])
    dynamic_sender_reciever_df['TimeSent'] =  pd.to_datetime(dynamic_sender_reciever_df['TimeSent'], format='%Y-%m-%d')
    dynamic_sender_reciever_df = dynamic_sender_reciever_df.iloc[40000:]
    dynamic_sender_reciever_df['From'] = dynamic_sender_reciever_df['From'].replace('houston <.ward@enron.com>', 'houston.ward@enron.com')
    #vocab_lookup = {str(idx):node for idx, node in enumerate(G.nodes())}
    #print(vocab_lookup)
    #dynnode2vec(vocab_lookup, adj, 1)
    #dynamic_sender_reciever_df['TimeSent'] =  pd.to_datetime(dynamic_sender_reciever_df['TimeSent'], format='%Y-%m-%d')

    #end_date = dynamic_sender_reciever_df['TimeSent'].iloc[-1]
    start_range = dynamic_sender_reciever_df['TimeSent'].iloc[0]

    end_range = start_range+pd.DateOffset(7)
    arr = []
    vocab_lookup_list = []
    vocab_lookup_dict = {}
    timestamp = []
    #while start_range <= end_date:
    for i in range(101):

        # filter the dataframe based on start and end range 
        filtered_df = dynamic_sender_reciever_df.loc[dynamic_sender_reciever_df['TimeSent'].between(start_range, end_range)]
        if filtered_df.empty:
            start_range = start_range+pd.DateOffset(8)
            end_range = start_range+pd.DateOffset(7) 
            continue

        filtered_df = filtered_df.groupby(by = ['From', 'To']).size().reset_index(name = 'counts')

        # create graph object out of filtered dataframe
        G = dataframe_to_graph_network(filtered_df, weight_arg)

        # create weighted/unweighted adjacency matrix from the graph object 
        adj = adjacency_dynamic_graph(G, weight_arg)
        timerange = [start_range, end_range]
        timestamp.append(timerange)
        # create a lookup dict that maps index to the node name while performing random walk 
        vocab_lookup_dict= {str(idx):node for idx, node in enumerate(G.nodes())}
        vocab_lookup_list.append(vocab_lookup_dict)
        arr.append(adj)

        # add 1 week time interval to start and end date 
        start_range = start_range+pd.DateOffset(8)
        end_range = start_range+pd.DateOffset(7)

    adj_matrix = np.asarray(arr)
    dynnode2vec(vocab_lookup_list, timestamp, adj_matrix, 2) 

def evolving_elliptic():

    arr = []
    vocab_lookup_list = []
    vocab_lookup_dict = {}
    timestamp = []
    edge_list = pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    node_features = pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header = None)
    node_classes = pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    node_features = node_features.iloc[:,0:2]
    node_features.columns = ['id', 'time']

    node_features = pd.merge(node_features, node_classes, left_on='id', right_on='txId', how='left')
    node_features = node_features[node_features['class']!='unknown']
    node_list = node_features['id'].tolist()
    filtered_edge_info = edge_list[edge_list['txId1'].isin(list(node_list))]
    filtered_edge_info = filtered_edge_info[filtered_edge_info['txId2'].isin(list(node_list))]

    filtered_edge_info = pd.merge(filtered_edge_info, node_features, left_on='txId1', right_on='id', how='left')

    for i in range(1, 50):
        edge_time_info = filtered_edge_info[filtered_edge_info['time'] == i]
         
        G = elliptic_dataframe_to_graph_network(edge_time_info)

        # create weighted/unweighted adjacency matrix from the graph object 
        adj = adjacency_dynamic_graph(G, weight_arg)
        timestamp.append(i)
        # create a lookup dict that maps index to the node name while performing random walk 
        vocab_lookup_dict= {str(idx):str(node) for idx, node in enumerate(G.nodes())}
        vocab_lookup_list.append(vocab_lookup_dict)
        arr.append(adj)

    adj_matrix = np.asarray(arr)
    dynnode2vec(vocab_lookup_list, timestamp, adj_matrix, 2)

def evolving_email_ham_spam():
    df = pd.read_pickle('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/email_ham_spam')
    sender_receiver_df = FilterDataset.create_sender_receiver_list_email(df, "dynamic")
    dynamic_sender_reciever_df = sender_receiver_df.sort_values(by = ['TimeSent'])
    print(dynamic_sender_reciever_df)
    dynamic_sender_reciever_df = dynamic_sender_reciever_df[20:-4]
    print(dynamic_sender_reciever_df)
    dynamic_sender_reciever_df['TimeSent'] =  pd.to_datetime(dynamic_sender_reciever_df['TimeSent'], format='%Y-%m-%d')
    #dynamic_sender_reciever_df = dynamic_sender_reciever_df.iloc[40000:]
    #dynamic_sender_reciever_df['From'] = dynamic_sender_reciever_df['From'].replace('houston <.ward@enron.com>', 'houston.ward@enron.com')
    #vocab_lookup = {str(idx):node for idx, node in enumerate(G.nodes())}
    #print(vocab_lookup)
    #dynnode2vec(vocab_lookup, adj, 1)
    #dynamic_sender_reciever_df['TimeSent'] =  pd.to_datetime(dynamic_sender_reciever_df['TimeSent'], format='%Y-%m-%d')

    end_date = dynamic_sender_reciever_df['TimeSent'].iloc[-1]
    start_range = dynamic_sender_reciever_df['TimeSent'].iloc[0]

    end_range = start_range+pd.DateOffset(7)
    print(start_range)
    print("End date", end_date)
    arr = []
    vocab_lookup_list = []
    vocab_lookup_dict = {}
    timestamp = []
    #while start_range <= end_date:
    for i in range(101):

        # filter the dataframe based on start and end range 
        filtered_df = dynamic_sender_reciever_df.loc[dynamic_sender_reciever_df['TimeSent'].between(start_range, end_range)]
        if filtered_df.empty:
            start_range = start_range+pd.DateOffset(8)
            end_range = start_range+pd.DateOffset(7) 
            continue

        filtered_df = filtered_df.groupby(by = ['From', 'To']).size().reset_index(name = 'counts')

        # create graph object out of filtered dataframe
        G = dataframe_to_graph_network(filtered_df, weight_arg)

        # create weighted/unweighted adjacency matrix from the graph object 
        adj = adjacency_dynamic_graph(G, weight_arg)
        timerange = [start_range, end_range]
        timestamp.append(timerange)
        # create a lookup dict that maps index to the node name while performing random walk 
        vocab_lookup_dict= {str(idx):node for idx, node in enumerate(G.nodes())}
        vocab_lookup_list.append(vocab_lookup_dict)
        arr.append(adj)

        # add 1 week time interval to start and end date 
        start_range = start_range+pd.DateOffset(8)
        end_range = start_range+pd.DateOffset(7)

    adj_matrix = np.asarray(arr)
    print(adj_matrix)
    print(vocab_lookup_list)
    print(timestamp)
    #dynnode2vec(vocab_lookup_list, timestamp, adj_matrix, 2) 

if __name__ == "__main__":
    weight_arg = sys.argv[1]
    dataset = sys.argv[2]
    if dataset == 'elliptic':
        evolving_elliptic()
    elif dataset == 'email_ham_spam':
        evolving_email_ham_spam()
    else:
        evolving_enron()
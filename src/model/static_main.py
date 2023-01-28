import sys
import pandas as pd
from graph import dataframe_to_graph_network, elliptic_dataframe_to_graph_network
from create_adjacency_matrix import get_adjacency_matrix, adjacency_dynamic_graph
from create_feat_matrix import get_feat_matrix
from create_feature_matrix_email_ham_spam import get_feat_matrix_email_ham_spam
from load_dataset import attribute_adjacency_matrix_utils, ellptic_dataset_adjacency_matrix_utils
from train_dominant import train_dominant, elliptic_train_dominant, email_spam_ham_train_dominant
from filterdataset import FilterDataset
import numpy as np
from numpy import count_nonzero
import networkx as nx
import matplotlib.pyplot as plt
from random import sample
from sklearn import preprocessing as p

def centrality_measure():
    # reading the dataset as pkl file
    df = pd.read_pickle('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/thesis_full_enron')

    #FilterDataset.count_max_sent_email(df)
    top_list, sender_receiver_df = FilterDataset.filter_df(df, weight_arg, "static")

    # create graph out of the above dataframe
    G = dataframe_to_graph_network(sender_receiver_df, weight_arg)
    print(len(G.nodes()))
    print(G.number_of_edges())
    
    '''key_list = []
    value_list = []
    for key, value in centrality.items():
        key_list.append(key)
        value_list.append(value)
    score_df = pd.DataFrame({'node':key_list,'score': value_list})
    score_df.sort_values(by = 'score', ascending=False, inplace = True)
    score_df.to_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/node_degree.csv')'''
    '''# remove low-degree nodes
    low_degree = [n for n, d in G.degree() if d < 300]
    G.remove_nodes_from(low_degree)

    components = nx.connected_components(G)
    largest_component = max(components, key=len)
    H = G.subgraph(largest_component)

    #centrality = H.degree()
    fig, ax = plt.subplots(figsize=(20, 15))

    pos = nx.spring_layout(H)
    #node_color = [community_index[n] for n in H]
    node_size = [d*200 for n, d in H.degree()]
    nx.draw_networkx(
        H,
        pos=pos,
        node_color='orange',
        with_labels=True,
        node_size=node_size,
        edge_color="gainsboro",
    )
    ax.text(
        0.80,
        0.06,
        "node size = Degree",
        horizontalalignment="center",
        transform=ax.transAxes,
    
    )

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.savefig('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/degree.png')'''

def enron_main():
    
    # reading the dataset as pkl file
    df = pd.read_pickle('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/thesis_full_enron')
    dir = r'/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/graphembedding/experiment/DFS_weighted_p_1_q_0.5_csv'
    '''read_embeddings = pd.read_csv(f'{dir}/88.csv')
    read_embeddings.rename(columns = {'Unnamed: 0': 'Sender'}, inplace = True)
    for column in read_embeddings.iloc[:,1:]:
    
        min = np.min(read_embeddings[column])
        max = np.max(read_embeddings[column])
        if max == 0:
            pass
        else:
            read_embeddings[column] = read_embeddings[column].apply(lambda x : ((x - min)/(max - min)))'''

    #FilterDataset.count_max_sent_email(df)
    top_list, sender_receiver_df = FilterDataset.filter_df(df, weight_arg, "static")

    # create graph out of the above dataframe
    G = dataframe_to_graph_network(sender_receiver_df, weight_arg)

    sender_list = []
    # filter the top_list of sender by adding the recipient's
    # of the sender in the sender's list
    for i in G.nodes:
        sender_list.append(i)
    sender_list = list(set(sender_list))

    # filter the dataframe based on top list
    df = df.loc[df['Sender'].isin(sender_list)]

    # ths function call is to create the feature matrix
    # from the graph and to return the node order list 
    #print(read_embeddings)
    feature_matrix, node_order_list = get_feat_matrix(df, sender_list)
    #feature_matrix = feature_matrix.merge(read_embeddings, how='left', left_on=feature_matrix.index, right_on = 'Sender')
    feature_matrix = feature_matrix.fillna(0)
    feature_matrix = feature_matrix.set_index('Sender')
    print(feature_matrix)
    # pass the graph in the below function
    # to create adjacency matrix
    adj = get_adjacency_matrix(G, node_order_list, weight_arg)
   
    # pass the feature matrix and adjacency matrix to the 
    # below function to get the normalized adjacency matrix
    adj, adj_label, attrs = attribute_adjacency_matrix_utils(adj, feature_matrix)
  
    # pass the normalized adjacency matrix and attribute matrix to 
    # the train_dominant function for training the model
    score = train_dominant(adj, adj_label, attrs)
    score_df = pd.DataFrame({'node':list(node_order_list),'score': score})
    score_df.sort_values(by = 'score', ascending=False, inplace = True)
    score_df.to_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/static_results/undirected_unweighted_dyn+stat_1.csv')

def elliptic_main():
    edge_info = pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    node_features = pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header = None)
    node_classes = pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')

    # renaming columns
    node_features.columns = ['id', 'time'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
    node_features = pd.merge(node_features, node_classes, left_on='id', right_on='txId', how='left')
    
    G = elliptic_dataframe_to_graph_network(edge_info)
    node_order_list = list(node_features['id'])
    adj_matrix = get_adjacency_matrix(G, node_order_list, weight_arg)
    truth = np.array(node_features['class']).flatten()
    adj, adj_label, attrs = ellptic_dataset_adjacency_matrix_utils(adj_matrix, node_features)
    score = elliptic_train_dominant(adj, adj_label, attrs, truth)
    #node_features = node_features[:100000]
    #  remove all the id's fromt the node_classes csv with class label as unknown
    '''labelled_node_classes = node_classes[node_classes['class'] != 'unknown']
    

    list_labelled_id = list(labelled_node_classes['txId'])
    
    edge_info = edge_info[edge_info['txId1'].isin(list(list_labelled_id))]
    #######edge_info = edge_info[edge_info['txId2'].isin(list(list_labelled_id))]
    G = elliptic_dataframe_to_graph_network(edge_info)
    #print("Number of nodes in the network", len(list(G.nodes)))

    final_node_list = list(G.nodes)
    final_class_df = node_classes[node_classes['txId'].isin(final_node_list) ]
    
    node_features = node_features[node_features['id'].isin(final_node_list)]
    ######labelled_node_classes = labelled_node_classes[labelled_node_classes['txId'].isin(final_node_list)]

    # merging class labels based on transaction id
    node_features = pd.merge(node_features, final_class_df, left_on='id', right_on='txId', how='left') '''

    '''#  filter only nodes without unknown class
    #filtered_node_features = node_features[node_features['class'] != 'unknown'][:20000]

    node_list = list(node_features['id'])
    filtered_edge_info = edge_info[edge_info['txId1'].isin(list(node_list))]
    filtered_edge_info = filtered_edge_info[filtered_edge_info['txId2'].isin(list(node_list))]

    filtered_node_classes = node_classes[node_classes['txId'].isin(list(node_list))]

    # merging class labels based on transaction id
    node_features = pd.merge(node_features, filtered_node_classes, left_on='id', right_on='txId', how='left')'''
    
    '''uncommon_node_list = []
    for node in edge_list:
        if node not in node_list:
            uncommon_node_list.append(node)
    node_feature_df = pd.concat([filtered_node_features, node_features[node_features['id'].isin(uncommon_node_list)]])
    node_feature_df.replace({'class' :{'2' : '0'}}, inplace = True)
    #node_feature_df.replace({'class': {'unknown':np.random.choice([1, 0], p = [0.3, 0.7])}}, inplace = True)
    np.random.seed(1) 
    m = node_feature_df['class'].eq('unknown')
    node_feature_df.loc[m, 'class'] = np.random.choice(['1', '0'], size=m.sum())
    node_feature_df.to_csv('elliptic_node_feature.csv')
    node_order_list = list(node_feature_df['id'])'''
    # This part of code os to include all thr nodes in the network
    '''G = elliptic_dataframe_to_graph_network(filtered_edge_info)
    print("Number of nodes in the graph", len(G.nodes()))
    node_order_list = list(node_features['id'])
    #node_features.replace({'class' :{'1' : 0}}, inplace = True)
    node_features.replace({'class' :{'2' : 0}}, inplace = True)
    node_features.replace({'class' :{'1' : 1}}, inplace = True)
    node_features.replace({'class' :{'unknown' : 0}}, inplace = True)'''
    
    '''node_features.replace({'class' :{'2' : 0}}, inplace = True)
    node_features.replace({'class' :{'1' : 1}}, inplace = True)
    node_features.replace({'class' :{'unknown' : 0}}, inplace = True)
    node_features = node_features.sample(frac=1)
    #print(node_features['class'].value_counts())
    node_order_list = list(node_features['id'])
    adj_matrix = get_adjacency_matrix(G, node_order_list, weight_arg)
    truth = np.array(node_features['class']).flatten()
    adj, adj_label, attrs = ellptic_dataset_adjacency_matrix_utils(adj_matrix, node_features)
    #print(attrs)
    #print(adj)
    #print(adj_label)
    #np.save('adj_label.npy', adj_label)
    #np.save('adj.npy',adj)
    elliptic_train_dominant(adj, adj_label, attrs, truth)'''

def elliptic_time():
    score_list = []
    time_list = []
    edge_list= pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    node_features = pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header = None)
    node_classes = pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    #dynamic_embedding = '/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/graphembedding/experiment/elliptic_BFS'
    #dynamic_embedding_df = pd.read_csv(f'{dynamic_embedding}/48.csv')
    #dynamic_embedding_df.rename(columns = {'Unnamed: 0' : 'txid'}, inplace = True)
    # renaming columns
    node_features.columns = ['id', 'time'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
    node_features = pd.merge(node_features, node_classes, left_on='id', right_on='txId', how='left')
    node_features = node_features[node_features['class']!='unknown']
    node_features.replace({'class' :{'2' : 0}}, inplace = True)
    node_features.replace({'class' :{'1' : 1}}, inplace = True)
    node_list = list(node_features['id'])
    filtered_edge_info = edge_list[edge_list['txId1'].isin(list(node_list))]
    filtered_edge_info = filtered_edge_info[filtered_edge_info['txId2'].isin(list(node_list))]

    #node_features = pd.merge(node_features, dynamic_embedding_df, left_on='id', right_on='txid', how='inner')
    #node_features.drop(columns = ['txid'], inplace = True)
    begin_time = 1
    while begin_time<=49:
        end_time = begin_time+48
        node_time_features = node_features[(node_features['time'] >= begin_time) &  (node_features['time'] <= end_time)]
        #node_time_features = node_features[node_features['time'] == begin_time]
        node_time_list = list(node_time_features['id'])
        time_edge_info = filtered_edge_info[filtered_edge_info['txId1'].isin(list(node_time_list))]
        time_edge_info = time_edge_info[time_edge_info['txId2'].isin(list(node_time_list))]
        
        G = elliptic_dataframe_to_graph_network(time_edge_info)
        #print(len(G.nodes()))
        #print(print(G.number_of_edges()))
        
        final_node_list = list(G.nodes())
        node_time_features = node_time_features[node_time_features['id'].isin(final_node_list)]

        node_order_list = list(node_time_features['id'])
        adj_matrix = get_adjacency_matrix(G, node_order_list, weight_arg)

        truth = np.array(node_time_features['class']).flatten()

        adj, adj_label, attrs = ellptic_dataset_adjacency_matrix_utils(adj_matrix, node_time_features)

        score = elliptic_train_dominant(adj, adj_label, attrs, truth)
        #score = float(score)
        #print(score)
        score_list.append(score)
        #time_list.append(str(begin_time)+'-'+str(end_time))
        #time_list.append(begin_time)
        #begin_time = begin_time+1   
        begin_time = end_time+1
    #df = pd.DataFrame(list(zip(time_list, score_list)),
               #columns =['Time', 'Score'])
    #df.to_csv("elliptic_undirected_individual_scores_0.5.csv")

def email_ham_spam():
    #df = pd.read_pickle('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/email_ham_spam')
    df = pd.read_pickle('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/email_ham_spam_with_hyperlinks')
    class_label = pd.read_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/email_spam_labels.csv')
    df = df[df['Sender'] != ""]
    class_label.dropna(inplace = True)
    #df = pd.merge(df, class_label, left_on='Sender', right_on='Sender', how='left')

    sender_receiver_df = FilterDataset.create_sender_receiver_list_email(df, "static")
    sender_receiver_df.dropna(inplace=True)
    
    # create graph out of the above dataframe
    G = dataframe_to_graph_network(sender_receiver_df, weight_arg)
    print('Number of nodes', len(G.nodes()))
    print('No of edges', G.number_of_edges())
    sender_list = list(G.nodes)
    #get_most_frequent_email_domain(node_list)

    df = df.loc[df['Sender'].isin(sender_list)]

    # ths function call is to create the feature matrix
    # from the graph and to return the node order list 

    feature_matrix, node_order_list = get_feat_matrix_email_ham_spam(df, sender_list)
    #feature_matrix.reset_index().merge(class_label, left_on = 'index', right_on = 'Sender', how="left").set_index('index')
    feature_matrix = feature_matrix.fillna(0)
    #print(feature_matrix)
    
    node_list_df = pd.DataFrame(node_order_list, columns=['Sender'])
    class_label = pd.merge(node_list_df, class_label, left_on='Sender', right_on='Sender', how='left')
    #class_label.replace({'label' :{1 : 0}}, inplace = True)
    class_label  = class_label.fillna(0)
    class_label.drop(columns = ['Unnamed: 0'], inplace = True)
    
    adj = get_adjacency_matrix(G, node_order_list, weight_arg)
    adj, adj_label, attrs = attribute_adjacency_matrix_utils(adj, feature_matrix)
    truth = np.array(class_label['label']).flatten()

    score = email_spam_ham_train_dominant(adj, adj_label, attrs, truth)
    score_df = pd.DataFrame({'node':list(node_order_list),'score': score})
    score_df.sort_values(by = 'score', ascending=False, inplace = True)
    score_df.to_csv('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/static_results/undirected_email_ham_spam.csv')

    
def get_most_frequent_email_domain(node_list):
    """calculates the most frequent domain address from the node list

    Args:
        G (_networkX graph_): graph network

    Returns:
        _str_: most frequent domain address
    """

    #node_list = list(G.nodes())
    domain_dict = {}
    for email_address in node_list:
        print("-----")
        try:
            domain = email_address.split('@')[1]
            if domain not in domain_dict:
                domain_dict[domain] = 1
            else:
                domain_dict[domain] += 1

        # If an email is a draft, then an empty space is encountered in
        # node_list as recipient
        except:
            continue
    ordered_domain_dict = sorted(domain_dict.items(), key=lambda kv: kv[1],
                                 reverse=True)
    domain_address = ordered_domain_dict[0][0]
    print(domain_address)
    return domain_address



if __name__ == "__main__":
    weight_arg = sys.argv[1]
    dataset = sys.argv[2]
    if dataset == 'elliptic':
        elliptic_main()
    elif dataset == 'elliptic_time':
        elliptic_time()
    elif dataset == 'email_ham_spam':
        email_ham_spam()
    else:
        enron_main()
        #centrality_measure()

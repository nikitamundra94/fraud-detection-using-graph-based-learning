import networkx as nx

def dataframe_to_graph_network(sender_receiver_df, weight_arg):
    # initialize the graph
    G = nx.Graph()

    if weight_arg == 'weighted':
        edge_attr = 'counts'
    else:
        edge_attr = None

    # Create networkx graph object from pandas dataframe
    G = nx.from_pandas_edgelist(sender_receiver_df, source="From", target="To", edge_attr = edge_attr)
    return G

def elliptic_dataframe_to_graph_network(edge_info):
    G = nx.from_pandas_edgelist(edge_info, source="txId1", target="txId2")

    return G
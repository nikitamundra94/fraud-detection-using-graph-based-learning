import networkx as nx

def get_adjacency_matrix(G, node_order_list, weight_arg):

    if weight_arg == 'weighted': 
        adjacency_matrix = nx.adjacency_matrix(G, nodelist = node_order_list, weight = 'counts').todense()
    else:
        adjacency_matrix = nx.adjacency_matrix(G, nodelist = node_order_list).todense()
        
    return adjacency_matrix

def adjacency_dynamic_graph(G, weight_arg):

    if weight_arg == 'weighted': 
        adjacency_matrix = nx.to_pandas_adjacency(G, weight = 'counts')
    else:
        adjacency_matrix = nx.to_pandas_adjacency(G)
        
    return adjacency_matrix
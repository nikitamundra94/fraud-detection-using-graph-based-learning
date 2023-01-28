from node2vec import Node2Vec
import numpy as np
import pandas as pd

def dynnode2vec(vocab_lookup_list, timestamp, graphs, epochs):
    tot_c = len(graphs) * epochs
    cnt = 0
    print(len(graphs))
    for j in range(epochs):
        
        #n2v = Node2Vec(vocab_lookup_list, graphs, dimensions=16, walk_length=4, num_walks=15, p=1, q=1, workers=8)
        #n2v.fit(save_wv_file='graphembedding/wv_0.emb', min_count=1, window=3)
        for i, graph in enumerate(graphs):
            graph = np.asarray(graph)
            cnt += 1
            print(f'Training node2vec model: {cnt} / {tot_c}')
            #print("list of vocab", vocab_lookup_list[i])
           
            n2v = Node2Vec(vocab_lookup_list[i], graph, dimensions=15, walk_length=3, num_walks=15, p=1, q=0.5, workers=8)
            if i == 0 and j == 0:
                m = n2v.fit(save_wv_file='graphembedding/wv_0.emb', min_count=1, window=3)
            elif i == 0:
                m = n2v.fit(initial_wv_file=f'graphembedding/wv_{len(graphs) - 1}.emb', save_wv_file=f'graphembedding/wv_{i}.emb', min_count=1,
                            window=2)
            else:
                m = n2v.fit(initial_wv_file=f'graphembedding/wv_{i - 1}.emb', save_wv_file=f'graphembedding/wv_{i}.emb', min_count=1,
                            window=2)

            if j == epochs-1:
                vocab, vectors = m.wv.vocab, m.wv.vectors
                name_index= np.array([(v[0], v[1].index) for v in vocab.items()])
                df =  pd.DataFrame(vectors[name_index[:,1].astype(int)])
                df.index = name_index[:, 0]
                df.sort_index(ascending=True, inplace=True)
                df.to_csv(f'graphembedding/experiment/email_ham_spam_DFS/{i}.csv')
                #df.to_csv(f'graphembedding/experiment/BFS_weighted_p_0.5_1_1/wv{timestamp[i][0]}_{timestamp[i][1]}_{i}.csv')
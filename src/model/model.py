import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import GraphConvolution
class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)

        # included for third gcn layer
        #self.gc3 = GraphConvolution(nhid,80)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        # included for third gcn layer
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc3(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)


        return x

class Attribute_decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_decoder, self).__init__()
        # to include gcn layer
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        #self.gc3 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = F.relu(self.gc2(x, adj))

        #x = F.dropout(x, self.dropout, training = self.training)
        #x = F.relu(self.gc3(x, adj))
        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        #self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
    
    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = x @ x.T
        
        return x

class Dominant(nn.Module):
        def __init__(self, feat_size, hidden_size, dropout):
            super(Dominant, self).__init__()
            self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
            self.attr_decoder = Attribute_decoder(feat_size, hidden_size, dropout)
            self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
        def forward(self, x, adj):
            # encode
            x = self.shared_encoder(x, adj)

            # decode Feature Matrix 
            x_hat = self.attr_decoder(x, adj)

            # decode Adjacency Matrix
            struct_reconst = self.struct_decoder(x, adj)
            
            return x_hat, struct_reconst
            # decode feature matrix
            #x_hat = self.attr_decoder(x, adj)
            # decode adjacency matrix
            #struct_reconstructed = self.struct_decoder(x, adj)
            # return reconstructed matrices
            #return struct_reconstructed, x_hat
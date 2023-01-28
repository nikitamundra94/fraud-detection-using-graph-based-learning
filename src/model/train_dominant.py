import torch
from model import Dominant
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
def loss_function(adj, A_hat, attrs, X_hat, alpha):

    # Feature reconstruct error
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attr_reconst_error = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attr_reconst_error)

    diff_adj = torch.pow(A_hat - adj, 2)
    adj_reconst_error = torch.sqrt(torch.sum(diff_adj,1))
    adj_cost = torch.mean(adj_reconst_error) 

    cost = alpha * attr_reconst_error + (1-alpha) * adj_reconst_error

    return cost, attribute_cost, adj_cost

def train_dominant(adj, adj_label, attrs):
    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)
    #print('Original Matrix', attrs)
    model = Dominant(feat_size = attrs.size(1), hidden_size = 12, dropout = 0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    for epoch in range(1,300):
        model.train()
        optimizer.zero_grad()
        x_hat, struct_reconstruct = model(attrs, adj)
        
        cost, struct_loss, feat_loss = loss_function(adj_label, struct_reconstruct, attrs, x_hat, 1)
        loss = torch.mean(cost)
        #print(loss)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            x_hat, struct_reconstruct = model(attrs, adj)
            for name, param in model.named_parameters():
                print(name, param)
            print('Reconstructed Matrix', x_hat)
            cost, struct_loss, feat_loss = loss_function(adj_label, struct_reconstruct, attrs, x_hat, 1)
            #score = cost.detach().numpy()
         
    score = cost.detach().numpy()

    return score

def elliptic_train_dominant(adj, adj_label, attrs, truth):
    lr = 0.005
    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)

    model = Dominant(feat_size = attrs.size(1), hidden_size = 100, dropout = 0.6)
  
    optimizer = torch.optim.Adam(model.parameters(), lr =lr)
    

    for epoch in range(51):
        model.train()
        optimizer.zero_grad()
        X_hat, struct_reconstruct = model(attrs, adj)
        loss, struct_loss, feat_loss = loss_function(adj_label, struct_reconstruct, attrs, X_hat, 0.5)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()        
        #print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))
        
        if epoch%10 == 0:
            model.eval()
            X_hat, struct_reconstruct = model(attrs, adj)
            #print(X_hat)
            loss, struct_loss, feat_loss = loss_function(adj_label, struct_reconstruct, attrs, X_hat, 0.5)
            score = loss.detach().cpu().numpy()
            
            #print(truth)
            #print(score)
            #print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(truth, score))
    np.save('score_undirected_full_elliptic.npy', score)
    np.save('truth_undirected_full_elliptic.npy', truth)
    #score = roc_auc_score(truth, score)
    #print('ROC:', score)
    return score

def email_spam_ham_train_dominant(adj, adj_label, attrs, truth):
    lr = 0.005
    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)

    model = Dominant(feat_size = attrs.size(1), hidden_size = 7, dropout = 0.6)
  
    optimizer = torch.optim.Adam(model.parameters(), lr =lr)
    

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        X_hat, struct_reconstruct = model(attrs, adj)
        loss, struct_loss, feat_loss = loss_function(adj_label, struct_reconstruct, attrs, X_hat, 0.7)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()        
        #print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))
        
        if epoch%50 == 0:
            model.eval()
            X_hat, struct_reconstruct = model(attrs, adj)
            #print(X_hat)
            #for name, param in model.named_parameters():
                #print(name, param)
            loss, struct_loss, feat_loss = loss_function(adj_label, struct_reconstruct, attrs, X_hat, 0.7)
            score = loss.detach().cpu().numpy()
            
            #print(truth)
            #print(score)
            #print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(truth, score))
    #np.save('score_final.npy', score)
    #np.save('truth_final.npy', truth)
    
    score = roc_auc_score(truth, score)
    print('ROC:', score)
    score = loss.detach().numpy()

    return score
    #return score

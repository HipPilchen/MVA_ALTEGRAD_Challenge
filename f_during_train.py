from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch
from torch import optim
import argparse
import time
import os
import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score
from torch.utils.tensorboard import SummaryWriter


dict_params = {'model_name':'distilbert-base-uncased', 'num_node_features':300, 'nout':200, 'nhid':300, 'graph_hidden_channels':300, 'drop': False, 'conv_layer': 'GCN'}
possible_model_names = ['allenai/scibert_scivocab_uncased', 'distilbert-base-uncased']

def main(args):
    
    """Compute Losses
    """
   
    CE = torch.nn.CrossEntropyLoss()
    
    def MSE(v1, v2):
        mse_loss = torch.nn.MSELoss(reduction='none')
        return mse_loss(v1, v2)
    

    def LS_loss(labels, graph_embed, text_embed, alpha):
        n = x_graph.size(0)  # batch-size
        
        # Compute cosine distance
        # distance =  1 - torch.matmul(F.normalize(graph_embed, p=2, dim=1),F.normalize(text_embed, p=2, dim=1).t())
        # Compute L2 distance        
        distance = torch.cdist(graph_embed,text_embed,p=2.0)
        loss = 0.0
           
        for i in range(n):
            for j in range(n):  
                    
                # Positive pairs
                if labels[i] == labels[j]:
                    
                    dist_pos = distance[i, j] # distance between positive pairs
                    
                    # Negative pairs
                    neg_contribution = torch.log(torch.sum(torch.exp(alpha-torch.abs(distance[i][labels != labels[j]])) 
                                                           + torch.exp(alpha-torch.abs(distance[j][labels != labels[i]]))))
                    
                    loss += F.relu(neg_contribution + dist_pos)**2

        return loss/2*n # normalize with number of positive pairs



        
    def contrastive_loss(v1, v2, temp=1, loss_type='basic', alpha=1):
       
        logits = torch.matmul(v1,torch.transpose(v2, 0, 1))/temp 
        labels = torch.arange(logits.shape[0], device=v1.device)
        
        if loss_type=='cos_loss':
            return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels) - torch.mean(torch.abs(torch.einsum('ij,ij->i', v1, v2))/(1e-9 + torch.einsum('ij,ij->i', v1, v1)*torch.einsum('ij,ij->i', v2, v2))) 
        elif loss_type=='mse':
            return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels) + MSE(v1, v2)
        elif loss_type=='basic':
            return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels) 
        elif loss_type=='lifted_structured_loss':     
            return LS_loss(labels,v1, v2, alpha)
        else: 
            print("Choose a loss")
        

    """Loads models and datasets
    """
    init_batch_size = args.init_bs
    final_batch_size = args.final_bs

    
    model_name = possible_model_names[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nb_epochs = args.num_epochs
    batch_size =   init_batch_size
    learning_rate = args.learning_rate

    val_loader = DataLoader(val_dataset, batch_size=init_batch_size, shuffle=True)
    
    dict_params['drop'] = args.drop
    dict_params['conv_layer'] = args.conv_layer
    dict_params['nout'] = args.nout
    dict_params['model_name'] = model_name
    model = Model(**dict_params) # nout = bert model hidden dim
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                    betas=(0.9, 0.999),
                                    weight_decay=args.weight_decay)

    name_exp = dict_params['conv_layer']  +'_modelname_'+str(args.model_name)+'_drop_' + str(dict_params['drop']) + '_temp_' + str(args.temp) + '_lr' + str(args.learning_rate) + '_losstype_'+ str(args.loss_type)+ '_alpha_' + str(args.alpha) + '_wd' + str(args.weight_decay) + '_initbs' + str(args.init_bs) + '_finalbs' + str(args.final_bs) + '_nout_' + str(args.nout)
    writer = SummaryWriter(comment=name_exp)
   
    epoch = 0
    loss = 0
    count_iter = 0
    time1 = time.time()
    printEvery = 50
    best_val_loss = 1000000
    best_lrap = 0
    previous_save = 0
    count_loss_decrease = 0
    n_update = -1
    batch_size = init_batch_size
    train_loader = DataLoader(train_dataset, batch_size=init_batch_size , shuffle=True)
    if model_name == 'allenai/scibert_scivocab_uncased':
        n_bert_layers = len(model.text_encoder.bert.encoder.layer)
    else:
        n_bert_layers = len(model.text_encoder.bert.transformer.layer)
       
    print('Model Parameters')  
    print('Dropout',dict_params['drop'])
    print('Loss','Losstype_'+ str(args.loss_type))
    print('Bert model',model_name)
    print('Number of bert layers: ', n_bert_layers)
    
    """Start training
    """
    for i in range(nb_epochs):
        epoch += 1
        print('-----EPOCH{}-----'.format(i+1))
        model.train()
        
        # Update batch size
        
        if i > 0 and i % args.n_ep_update == 0:  
            if n_bert_layers <= n_update < n_bert_layers + 4:
                n_update += 1
                # Tests different batchsize increase policies
                # batch_size = int(init_batch_size*(n_bert_layers+4-n_update)/(n_bert_layers+4) + n_update*final_batch_size/(n_bert_layers+4))
            elif n_update < n_bert_layers :
                n_update += 1
                # batch_size += 4
                
        if i%2==0 and batch_size < final_batch_size:
            batch_size += 4
             
        # Freeze layers (first BERT then CONV layers)     
        if n_update >= 0 and n_update < n_bert_layers:
            if model_name == 'allenai/scibert_scivocab_uncased':
                for layer in model.text_encoder.bert.encoder.layer[:n_update]:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                for layer in model.text_encoder.bert.transformer.layer[:n_update]:
                    for param in layer.parameters():
                        param.requires_grad = False
            
                    
        elif n_update == n_bert_layers+1:
            for param in model.graph_encoder.conv1.parameters():
                param.requires_grad = False
                
        elif n_update == n_bert_layers+2:
            for param in model.graph_encoder.conv2.parameters():
                param.requires_grad = False
        
        elif n_update == n_bert_layers+3:
            for param in model.graph_encoder.conv3.parameters():
                param.requires_grad = False
        
        elif n_update == n_bert_layers+4:
            for param in model.graph_encoder.attention.parameters():
                param.requires_grad = False
        
 
       
        train_loss = []          
        train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle=True)  
        print('Current batch size: ', batch_size)
        for (batch_idx, batch) in enumerate(train_loader):
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            try:            
                x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            
            except IndexError:
                print('Index error')
                continue
            except torch.cuda.OutOfMemoryError:
                print('Batch_size error')
                batch_size -=  10 
                break 
            
            current_loss = contrastive_loss(x_graph, x_text,temp = args.temp, loss_type=args.loss_type, alpha=args.alpha)  
            current_loss.backward()
            optimizer.step()  
            optimizer.zero_grad()    
            loss += current_loss.item()
            
            
            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/printEvery))
                writer.add_scalar("Losses/Train",loss/printEvery,count_iter)
                writer.add_scalar("Time",time2,count_iter)
                
                train_loss.append(loss/printEvery)
                loss = 0 
                
        # Model evaluation
        model.eval()       
        val_loss = 0        
        for batch in val_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            
            try:
                x_graph, x_text = model(graph_batch.to(device), 
                                        input_ids.to(device), 
                                        attention_mask.to(device))
            except IndexError:
                print('Index error')
                continue
            
            current_loss = contrastive_loss(x_graph, x_text,temp = args.temp, loss_type=args.loss_type, alpha=args.alpha)   
            val_loss += current_loss.item()

        # Store parameters 
        lrap = compute_lraps(model, val_dataset, val_loader, device)
        best_lrap = max(best_lrap, lrap)
        writer.add_scalar('Losses/Train per epoch',np.mean(np.array(train_loss)),epoch)
        writer.add_scalar('Losses/Validation',val_loss,epoch)
        writer.add_scalar('Losses/LRAP',lrap,epoch)
        best_val_loss = min(best_val_loss, val_loss)
        print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)),'LRAP: ',lrap )

        # Load weights and check if validation/LRAP loss improved
        if best_lrap==lrap:
            
            print('validation loss improoved saving checkpoint...')
            if i >0:
                os.remove(os.path.join('./weights/', name_exp+'_model'+str(previous_save)+'.pt'))
                
            save_path = os.path.join('./weights/', name_exp+'_model'+str(i)+'.pt')
            previous_save = i
            
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'lrap': lrap
            }, save_path)
            
            print('checkpoint saved to: {}'.format(save_path))
            print('LRAP: ', lrap)
           
            count_loss_decrease = 0
            
        elif best_val_loss == val_loss:
            count_loss_decrease = 0        
        else :
            count_loss_decrease += 1
            if count_loss_decrease == 10:
                print('Training stopped at %i epochs'%i)
                break
            
    writer.flush()
    writer.close()


    """Generate submission file from best model.
    """
    
    print('loading best model...')
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()

    test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(test_cids_dataset, batch_size=init_batch_size, shuffle=False)

    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=init_batch_size, shuffle=False)
    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())


    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    solution.to_csv(name_exp+'_submission.csv', index=False)
    
    """Compute LRAP during training
    """

def compute_lraps(model, val_dataset, val_loader, device):
    graph_embeddings = []
    text_embeddings = []
    count = 0
    
    for batch in val_loader:
        if count == 200:
            break
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        try:
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
        except IndexError:
                print('Index error')
                continue
        except torch.cuda.OutOfMemoryError:
                print('Batch_size error')
                batch_size -=  10 
                break 
        graph_embeddings.append(x_graph.tolist())
        text_embeddings.append(x_text.tolist())
        count += 1

    labels = np.eye(len(val_dataset))
    
    graph_embeddings = [item for sublist in graph_embeddings for item in sublist]
    text_embeddings = [item for sublist in text_embeddings for item in sublist]
    
    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    lrap = label_ranking_average_precision_score(labels, similarity)
    return lrap
    
"""Parser arguments
"""
def parser_args(parser):

    parser.add_argument('-n', '--num-epochs', default=500, type=int,
                        help="number of epochs to run") 
    
    parser.add_argument('-lr', '--learning-rate', default=2e-5, type=float,
                        help="learning rate for Adam optimizer")

    parser.add_argument('-conv_layer','--conv_layer', type=str, default='GCN',
                        help='Type of convolutional layers')
    
    parser.add_argument('-n_ep_update','--n_ep_update', type = int, default = 10, help='How many epoch between each')
    
    parser.add_argument('-init_bs','--init_bs', type=int, default=100,
                        help='Batchsize when nothing freezed')
    
    parser.add_argument('-final_bs','--final_bs', type=int, default=200,
                        help='Batchsize when 2 Convs and Bert freezed')
    
    parser.add_argument('-temp','--temp',type=float, default = 1., help = 'Temperature which divide logits in the Cross Entropy Loss')
    
    parser.add_argument('-wd','--weight_decay',type=float, default = 0.01, help = 'Weight decay')
    
    parser.add_argument('-dropout','--drop',type=bool, default = False, help = 'Dropout between linear dense layers')

    parser.add_argument('-loss_type','--loss_type',type=str, default ='basic', help = 'Choose the type of loss')
    
    parser.add_argument('-alpha', '--alpha', type=float, default=1., help='')
    
    parser.add_argument('-nout', '--nout', type=int, default=200)
    
    parser.add_argument('-model_name','--model_name',type=int, default = 1, help = 'Index in the list for Text encoder model name')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser) 
    args = parser.parse_args()

    main(args)
    
    

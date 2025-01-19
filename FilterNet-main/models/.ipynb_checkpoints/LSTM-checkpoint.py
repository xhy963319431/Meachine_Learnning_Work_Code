import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
    
        self.revin_layer = RevIN(configs.enc_in , affine = True , subtract_last = False)
        
        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size
        
        self.lstm = nn.LSTM(self.seq_len , self.hidden_size ,batch_first=True )
        self.linear = nn.Linear(self.hidden_size, self.pred_len)
    
    
    def forward(self, x , x_mark_enc , x_dec , x_mark_dec , mask = None ):
        # x: [Batch, Input length, Channel]
        z = x
        z = self.revin_layer(z , 'norm')
        x = z
        x = x.permute(0, 2, 1) # x , B , C , I
        
        h0 = torch.zeros(1 , x.size(0) , self.hidden_size ).to(x.device)
        c0 = torch.zeros(1 , x.size(0) ,self.hidden_size).to(x.device)
        out, _ = self.lstm(x , ( h0 ,  c0) )
        x = self.linear(out)
        # print(x.shape)
        
        x = x.permute(0, 2, 1)
        
        
        
        z = x
        z = self.revin_layer( z, 'denorm' )
        x = z
        
        return x
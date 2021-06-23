import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    Encoder.
    """
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

# TODO : MODIFY THE DECODER.
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # define the properties
        self.embed = embed_size
        self.hidden_size =hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
         # Embedding Layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM Cell
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first= True,
                           dropout = 0)
    
        #Output Fully Connected Layer
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        #activations
        self.softmax=nn.LogSoftmax(dim = 1)

    def forward(self, features, captions):
        # Embed the features vector from CNN into embedding
        # Caption shape will be ex) ([10,16] tensor.)
        captions = self.embed(captions[:,:-1])
        
        # concatenate into embeded vectors
        embeddings = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        # Push it through lstm layers
        lstm_out, self.hidden = self.lstm(embeddings)
        
        # Lastly, linear layer.
        outputs = self.fc_out(lstm_out)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_output = []
        
        for i in range(max_len):
            hidden, states = self.lstm(inputs, states)
            outputs = self.fc_out(hidden.squeeze(1))
            _, predicted = outputs.max(dim=1)
            predicted_output.append(predicted.item())
            
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        
        return predicted_output
        
# To imitate the behavior described using torch.optim.lr_scheduler.CosineAnnealingLR, you need to carefully adjust its parameters. However, the behavior you're describing is a custom exponential decay rather than a cosine annealing schedule. To simulate this using PyTorch, you may want to use a custom learning rate scheduler instead.
# Here’s how you can implement the described behavior:

# 1. Using torch.optim.lr_scheduler.LambdaLR
# The LambdaLR scheduler allows you to define a custom learning rate schedule as a lambda function.
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, x_input,labels, seq_length, batch_size):
        self.x_input = x_input
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_batches = len(x_input) // seq_length
        self.stride = len(x_input) // batch_size  # How far apart starting points are for batches
        self.labels = labels


    def __len__(self):
        return self.num_batches  # One sequence per batch

    def __getitem__(self, idx):
        # print("__getitem__ call number: ", idx)
        # Calculate the start of the sequence for this batch
        start_idx = (((idx%self.batch_size)) * self.stride) + (idx//batch_size)*(self.seq_length)
        # Get the sequence
        input_sequence = self.x_input[start_idx:start_idx + self.seq_length]
        label_sequence = self.labels[start_idx:start_idx + self.seq_length]
        sequence = (input_sequence, label_sequence)

        return sequence

def lr_lambda(epoch):
    if epoch <= 6:
        return 1.0  # No change for the first 6 epochs
    else:
        return (1.0 / (1.5 ** (epoch - 6)))  # Decrease by factor of 1.5 per epoch after epoch 6

class hw2LSTM(nn.Module):

    # reason: in the Penn Treebank (PTB) dataset, it is common to treat the text
    # as a single continuous stream of words when training language models

    def __init__(self, batch_size,seq_length, hidden_dim, n_layers, vocab_size, embedding_dim, dropout_p, device, rnn_type = "LSTM",checkpoint = None):
        super(hw2LSTM, self).__init__()  # Pass the current class and self
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_p, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout_p, batch_first=True)

        self.fc = nn.Linear(hidden_dim,vocab_size)

        self.apply(self.init_weights)
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.device = device
        self.rnn_type = rnn_type

        if checkpoint is not None:
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint)
            self.to(device)
            print("Checkpoint loaded successfully.")



    def init_weights(self,m):
        # LSTM parameters are initialized uniformly in [−0.05, 0.05].
         if type(m) == nn.LSTM or type(m) == nn.GRU:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.uniform_(param.data, -0.05, 0.05)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0) # convention


    def forward(self,x,hidden):
        embedded = self.embedding(x) # batch_size*seq_length,embedding
        embedded = self.dropout(embedded)
        out, hidden = self.rnn(embedded, hidden) # batch_size, seq_length, hidden_dim
        out = out.contiguous().view(-1,self.hidden_dim) # batch_size*seq_length , hidden_dim
        out = self.fc(out) # batch_size*seq_length , vocab_size
        return out, hidden

def lr_lambda(epoch):
    if epoch <= 6:
        return 1.0  # No change for the first 6 epochs
    else:
        return (1.0 / (1.2 ** (epoch - 6)))  # Decrease by factor of 1.2 per epoch after epoch 6

def init_hidden(net, batch_size, n_layers, hidden_dim):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # hidden0 should tuple each in size (n_layers,batchsize,hidden_dim)
    weight = next(net.parameters()).data

    # If theres multiple seqs per batch
    if net.rnn_type == "LSTM":
    # Initialize hidden state as a tuple (h_t, c_t)
        hidden = (
        weight.new_zeros(n_layers, batch_size, hidden_dim).to(device),
        weight.new_zeros(n_layers, batch_size, hidden_dim).to(device),
      )
    elif net.rnn_type == "GRU":
    # Initialize hidden state as a single tensor
        hidden = weight.new_zeros(n_layers, batch_size, hidden_dim).to(device)

    return hidden

def train(model,train_x,train_y,val_x,val_y,batch_size,learning_rate, n_epoches, clip,validate_and_print_every,model_name):
    count = 0
    best_valid_perplexity = float('inf')

    # checkpoints
    checkpoint_dir = f'Perplexity_Comparasion_RNN/models/checkpoints/{model_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Zero accumulated gradients
    model.zero_grad()

    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)


    train_dataset = SequenceDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), model.seq_length, batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # use device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    # Loss & Perplexity
    per_epoch_train_losses = []
    per_epoch_train_perplexities = []
    per_epoch_valid_losses = []
    per_epoch_valid_perplexities = []

    # epoch_loop
    for epoch in tqdm(range(n_epoches)):
      model.train()
      valid_losses = []
      valid_perplexities = []

      # batch_loop
      # Initiate hidden once. Afterwards - the final hidden states of the current minibatch
      # is used as the initial hidden state of the subsequent minibatch
      hidden = init_hidden(model,batch_size,model.n_layers,model.hidden_dim)

      # batch_loop
      for inputs, labels in train_loader:
          count += 1

          # Throw last batch
          # if inputs.shape[0] != batch_size or labels.shape[0] != batch_size:
          #       continue  # Skip this batch if it's smaller

          model.zero_grad()


          inputs, labels = inputs.to(device), labels.to(device)
          if model.rnn_type == "LSTM":
            hidden = tuple([each.detach() for each in hidden])
          elif model.rnn_type == "GRU":
            hidden = hidden.detach()
          out, hidden = model(inputs,hidden)
          labels = labels.view(-1)
          loss = criterion(out, labels)
          loss.backward()
          # We clip the norm of the gradients (normalized by minibatch size) at 5.
          nn.utils.clip_grad_norm_(model.parameters(), clip)
          # Note - I couldnt find if clip_grad_norm_ normalizes the gradients internally by batch_size.
          optimizer.step()

          if count % validate_and_print_every == 0:
            ###
            train_perplexity = torch.exp(loss).item()
            ###
            print("Epoch: {}/{}...".format(epoch+1, n_epoches),
                "Step: {}...".format(count),
                "Loss: {:.4f}...".format(loss.item()),
                "Last train batch Perplexity: {:.4f}...".format(train_perplexity))
                # "Avarage val Loss: {:.4f}...".format(valid_loss/len(val_loader)),
                # "Avarage val Perplexity: {:.4f}".format(average_valid_perplexity))

      # Validation
      model.eval()

      valid_losses = []
      valid_perplexities_avarage = []
      val_h = init_hidden(model,batch_size,model.n_layers,model.hidden_dim)
      val_dataset = SequenceDataset(torch.from_numpy(val_x), torch.from_numpy(val_y), model.seq_length, batch_size)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

      # Iterate throught validation data
      for inputs, labels in val_loader:
          # if inputs.shape[0] != batch_size or labels.shape[0] != batch_size:
          #     continue
          if model.rnn_type == "LSTM":
            val_h = tuple([each.detach() for each in val_h])
          elif model.rnn_type == "GRU":
            val_h = val_h.detach()
          inputs, labels = inputs.to(device), labels.to(device)
          out, val_h = model(inputs, val_h)
          labels = labels.view(-1)
          val_loss = criterion(out, labels)
          valid_losses.append(val_loss.item())

          valid_perplexity = torch.exp(val_loss).item()
          valid_perplexities_avarage.append(valid_perplexity)

      # valid_losses.append(valid_loss/len(val_loader))
      avarage_valid_loss = sum(valid_losses)/len(valid_losses)
      average_valid_perplexity = sum(valid_perplexities_avarage)/len(valid_perplexities_avarage)
      valid_perplexities.append(average_valid_perplexity)

      # Iterate through train data
      epoch_train_losses = []
      epoch_train_perplexities = []
      hidden = init_hidden(model,batch_size,model.n_layers,model.hidden_dim)
      for inputs, labels in train_loader:
          if model.rnn_type == "LSTM":
            hidden = tuple([each.detach() for each in hidden])
          elif model.rnn_type == "GRU":
            hidden = hidden.detach()
          inputs, labels = inputs.to(device), labels.to(device)
          out, hidden = model(inputs, hidden)
          labels = labels.view(-1)
          loss = criterion(out, labels)
          epoch_train_losses.append(loss.item())
          epoch_train_perplexity = torch.exp(loss).item()
          epoch_train_perplexities.append(epoch_train_perplexity)

      avarage_train_loss = sum(epoch_train_losses)/len(epoch_train_losses)
      avarage_train_perplexity = sum(epoch_train_perplexities)/len(epoch_train_perplexities)
      model.train()

      if average_valid_perplexity < best_valid_perplexity:
        best_valid_perplexity = average_valid_perplexity
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_validation_model.pt'))

      tqdm.write(
                f'Epoch: {epoch + 1:02}, Train Loss: {avarage_train_loss:.3f}, Valid Loss: {avarage_valid_loss:.3f}, Train Perplexity: {avarage_train_perplexity:.3f}, Valid Perplexity: {average_valid_perplexity:.3f}')

      per_epoch_train_losses.append(avarage_train_loss)
      per_epoch_train_perplexities.append(avarage_train_perplexity)
      per_epoch_valid_losses.append(avarage_valid_loss)
      per_epoch_valid_perplexities.append(average_valid_perplexity)
    return per_epoch_train_losses, per_epoch_valid_losses, per_epoch_train_perplexities, per_epoch_valid_perplexities

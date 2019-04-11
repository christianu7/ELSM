import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim



class Model(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, embedding_dim=8, num_layers=1):
        """
            __init__(Model, int, int, int) -> None
            Implements the model without splitting with a Bi-Directional LSTM
            and no neural network for computing output. Output matrix is binary.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            num_layers: Number of layers in the LSTM
        """
        super(Model, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for mean and variance
        self.mean = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(Model) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def forward(self, A):
        """
            forward(Model, ndarray) -> (ndarray, ndarray, ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            
            Returns:
                means: (input_dim, embedding_dim, time_steps) Mean for each
                            node for each time step 
                log_vars: (input_dim, embedding_dim, time_steps) Log 
                          variance for each node for each time step
                embeddings: (input_dim, embedding_dim, time_steps) Embeddings
                            sampled from each node
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Get the means, log-variance, a_preds and embeddings
        means = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        log_vars = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        embeddings = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        
        means = Variable(means)
        log_vars = Variable(log_vars)
        embeddings = Variable(embeddings)
        
        for i in range(num_steps):
            means[:, :, i] = self.mean(out[i, :, :])
            log_vars[:, :, i] = self.log_var(out[i, :, :])
            
            embeddings[:, :, i] = means[:, :, i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                  torch.sqrt(torch.exp(log_vars[:, :, i]))
        
        return (means, log_vars, embeddings)
        

    
    
    def loss_function(self, A, embeddings, variances, m=0.0, s=1.0, \
                                                        s1=1.0, s2=1.0, s4=1.0):
        """
            loss_function(Model, ndarray, ndarray, ndarray, ndarray, float, \
                                                   float, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) Variational distri-
                        -bution variances
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
               vector is used
            s: Scale parameter for prior on w^(1)
            s1: Scale parameter for connections P(A|Z)
            s2: Scale parameter for the influence
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
            
        # Compute log p(z^(1))
        log_pz1 = -torch.sum((embeddings[:, :, 0] - m) ** 2) / (2 * s**2)
        
        # Compute distances between embeddings
        distances = Variable(torch.zeros(n, n, T))
        for t in range(T):
            dist = embeddings[:, :, t].unsqueeze(0).expand(n, n, d)
            dist = torch.sum((dist - dist.permute(1, 0, 2))**2, dim=2)
            distances[:, :, t] = dist
        
        # Compute prod_t P(A^(t) | z^(t))
        a_preds = 1 - torch.tanh(distances / (s1**2))
        log_a_z = torch.triu((A * torch.log(1e-6 + a_preds) + (1-A) * torch.log(\
                           1e-6 + 1 - a_preds)).sum(dim=2), diagonal=1).sum()
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1))
        d2 = torch.exp(-distances / (s2**2))
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1] + Variable( \
                    torch.eye(n)))).sum(dim=1)).unsqueeze(1).expand(n, d)
            log_z_az -= torch.sum((embeddings[:, :, t] - means) ** 2) / (2 * s4**2)
        
        # Compute the entropy term
        ent = torch.sum(variances) / 2
        
        # Compute ELBO
        ELBO = log_pz1 + log_a_z + log_z_az + ent
        
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, s1=1.0, s2=1.5):
        """
            gen_next_layer(Model, ndarray, ndarray, float, float) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            s1: Scale paramter for output predictions
            s2: Scale parameter for influence
            
            Returns:
                out: The predicted output probabilities for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-dist / s2 ** 2)
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)) ** 2).sum(dim=2)
        a_preds = 1 - torch.tanh(dist2 / (s1**2))
        
        return a_preds
        
        
        
        
        
        
        
        
        
class ModelNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, embedding_dim=8, num_layers=1):
        """
            __init__(ModelNN, int, int, int) -> None
            Implements the model without splitting with a Bi-Directional LSTM.
            Uses neural network for output predictions. Output matrix is binary.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            num_layers: Number of layers in the LSTM
        """
        super(ModelNN, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for mean and variance
        self.mean = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        
        # Set up the output prediction layer
        self.a_pred = nn.Linear(self.embedding_dim, 1)
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelNN) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def forward(self, A):
        """
            forward(ModelNN, ndarray) -> (ndarray, ndarray, ndarray, ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            
            Returns:
                means: (input_dim, embedding_dim, time_steps) Mean for each
                            node for each time step 
                log_vars: (input_dim, embedding_dim, time_steps) Log 
                          variance for each node for each time step
                embeddings: (input_dim, embedding_dim, time_steps) Embeddings
                            sampled from each node
                a_preds: Predictions for the output as given by NN
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Get the means, log-variance, a_preds and embeddings
        means = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        log_vars = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        embeddings = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        a_preds = torch.Tensor(self.input_dim, self.input_dim, num_steps)
        
        means = Variable(means)
        log_vars = Variable(log_vars)
        embeddings = Variable(embeddings)
        a_preds = Variable(a_preds)
        
        for i in range(num_steps):
            means[:, :, i] = self.mean(out[i, :, :])
            log_vars[:, :, i] = self.log_var(out[i, :, :])
            
            embeddings[:, :, i] = means[:, :, i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                  torch.sqrt(torch.exp(log_vars[:, :, i]))
                  
            dist = embeddings[:, :, i].unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
            dist = (dist - dist.permute(1, 0, 2)).view(-1, \
                                            self.embedding_dim) ** 2
            a_preds[:, :, i] = F.sigmoid(self.a_pred(dist).view(\
                                      self.input_dim, self.input_dim))
        
        return (means, log_vars, embeddings, a_preds)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, m=0.0, s=1.0, \
                                                                s2=1.0, s4=1.0):
        """
            loss_function(ModelNN, ndarray, ndarray, ndarray, ndarray, ndarray,\
                                                   float, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) Variational distri-
                        -bution variances
            a_preds: (input_dim, input_dim, num_steps) Output predictions from
                     the neural network
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
               vector is used
            s: Scale parameter for prior on w^(1)
            s2: Scale parameter for the influence
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
            
        # Compute log p(z^(1))
        log_pz1 = -torch.sum((embeddings[:, :, 0] - m) ** 2) / (2 * s**2)
        
        # Compute distances between embeddings
        distances = Variable(torch.zeros(n, n, T))
        for t in range(T):
            dist = embeddings[:, :, t].unsqueeze(0).expand(n, n, d)
            dist = torch.sum((dist - dist.permute(1, 0, 2))**2, dim=2)
            distances[:, :, t] = dist
        
        # Compute prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((A * torch.log(1e-6 + a_preds) + (1-A) * torch.log(\
                           1e-6 + 1 - a_preds)).sum(dim=2), diagonal=1).sum()
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1))
        d2 = torch.exp(-distances / (s2**2))
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1] + Variable( \
                    torch.eye(n)))).sum(dim=1)).unsqueeze(1).expand(n, d)
            log_z_az -= torch.sum((embeddings[:, :, t] - means) ** 2) / (2 * s4**2)
        
        # Compute the entropy term
        ent = torch.sum(variances) / 2
        
        # Compute ELBO
        ELBO = log_pz1 + log_a_z + log_z_az + ent
        
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, s2=1.5):
        """
            gen_next_layer(ModelNN, ndarray, ndarray, float) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            s2: Scale parameter for influence
            
            Returns:
                out: The predicted output probabilities for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-dist / s2 ** 2)
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = (dist2 - dist2.permute(1, 0, 2)).view(-1, self.embedding_dim)**2
        a_preds = F.sigmoid(self.a_pred(dist2).view(\
                                      self.input_dim, self.input_dim))
        
        return a_preds
        
        
        
        
        
        
        
        

class ModelNNScale(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, embedding_dim=8, num_layers=1,\
                    use_bias=False):
        """
            __init__(ModelNNScale, int, int, int, bool) -> None
            Implements the model without splitting with a Bi-Directional LSTM.
            Uses neural network for predicting scale parameter s1 in output
            prediction. Output matrix is binary.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            num_layers: Number of layers in the LSTM
            use_bias: Whether to use bias term in output scale calculation
        """
        super(ModelNNScale, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for mean and variance
        self.mean = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        
        # Set up the output prediction layer
        self.a_pred = nn.Linear(1, 1, bias=use_bias)
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelNNScale) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def forward(self, A):
        """
            forward(ModelNNScale, ndarray) -> (ndarray, ndarray, ndarray, \
                                                                        ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            
            Returns:
                means: (input_dim, embedding_dim, time_steps) Mean for each
                            node for each time step 
                log_vars: (input_dim, embedding_dim, time_steps) Log 
                          variance for each node for each time step
                embeddings: (input_dim, embedding_dim, time_steps) Embeddings
                            sampled from each node
                a_preds: Predictions for the output as given by NN
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Get the means, log-variance, a_preds and embeddings
        means = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        log_vars = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        embeddings = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        a_preds = torch.Tensor(self.input_dim, self.input_dim, num_steps)
        
        means = Variable(means)
        log_vars = Variable(log_vars)
        embeddings = Variable(embeddings)
        a_preds = Variable(a_preds)
        
        for i in range(num_steps):
            means[:, :, i] = self.mean(out[i, :, :])
            log_vars[:, :, i] = self.log_var(out[i, :, :])
            
            embeddings[:, :, i] = means[:, :, i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                  torch.sqrt(torch.exp(log_vars[:, :, i]))
                  
            dist = embeddings[:, :, i].unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
            dist = ((dist - dist.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
            a_preds[:, :, i] = 1 - F.tanh(torch.abs(self.a_pred(dist).view(\
                                      self.input_dim, self.input_dim)))
            
        return (means, log_vars, embeddings, a_preds)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, m=0.0, s=1.0, \
                                                                s2=1.0, s4=1.0):
        """
            loss_function(ModelNNScale, ndarray, ndarray, ndarray, ndarray, \
                                          ndarray, float, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) Variational distri-
                        -bution variances
            a_preds: (input_dim, input_dim, num_steps) Output predictions from
                     the neural network
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
               vector is used
            s: Scale parameter for prior on w^(1)
            s2: Scale parameter for the influence
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
            
        # Compute log p(z^(1))
        log_pz1 = -torch.sum((embeddings[:, :, 0] - m) ** 2) / (2 * s**2)
        
        # Compute distances between embeddings
        distances = Variable(torch.zeros(n, n, T))
        for t in range(T):
            dist = embeddings[:, :, t].unsqueeze(0).expand(n, n, d)
            dist = torch.sum((dist - dist.permute(1, 0, 2))**2, dim=2)
            distances[:, :, t] = dist
        
        # Compute prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((A * torch.log(1e-6 + a_preds) + (1-A) * torch.log(\
                           1e-6 + 1 - a_preds)).sum(dim=2), diagonal=1).sum()
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1))
        d2 = torch.exp(-distances / (s2**2))
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1] + Variable( \
                    torch.eye(n)))).sum(dim=1)).unsqueeze(1).expand(n, d)
            log_z_az -= torch.sum((embeddings[:, :, t] - means) ** 2) / (2 * s4**2)
        
        # Compute the entropy term
        ent = torch.sum(variances) / 2
        
        # Compute ELBO
        ELBO = log_pz1 + log_a_z + log_z_az + ent
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, s2=1.5):
        """
            gen_next_layer(ModelNNScale, ndarray, ndarray, float) -> \
                                                                        ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            s2: Scale parameter for influence
            
            Returns:
                out: The predicted output probabilities for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-dist / s2 ** 2)
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
        a_preds = 1 - F.tanh(torch.abs(self.a_pred(dist2).view(\
                                      self.input_dim, self.input_dim)))
        
        return a_preds










class ModelNNScale2(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, embedding_dim=8, num_layers=1,\
                    use_bias_s1=False, use_bias_s2=False):
        """
            __init__(ModelNNScale2, int, int, int, bool) -> None
            Implements the model without splitting with a Bi-Directional LSTM.
            Uses neural network for predicting scale parameter s1 in output
            prediction. Output matrix is binary. Also provides scaled version
            of distances between embeddings eliminating the need for manually
            setting scale parameter in computing influence.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            num_layers: Number of layers in the LSTM
            use_bias_s1: Whether to use bias in output scale calculation
            use_bias_s1: Whether to use bias in influence scale calculation
        """
        super(ModelNNScale2, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for mean and variance
        self.mean = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        
        # Set up the output prediction layer
        self.a_pred = nn.Linear(1, 1, bias=use_bias_s1)

        # Set up scaled distance layer
        self.scaled_dist = nn.Linear(1, 1, bias=use_bias_s2)
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelNNScale2) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def forward(self, A):
        """
            forward(ModelNNScale2, ndarray) -> (ndarray, ndarray, ndarray, \
                                                                        ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            
            Returns:
                means: (input_dim, embedding_dim, time_steps) Mean for each
                            node for each time step 
                log_vars: (input_dim, embedding_dim, time_steps) Log 
                          variance for each node for each time step
                embeddings: (input_dim, embedding_dim, time_steps) Embeddings
                            sampled from each node
                a_preds: Predictions for the output as given by NN
                scaled_dists: (input_dim, input_dim, time_steps) Scaled dist-
                               -tances between embeddings for influence
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Get the means, log-variance, a_preds and embeddings
        means = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        log_vars = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        embeddings = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        a_preds = torch.Tensor(self.input_dim, self.input_dim, num_steps)
        scaled_dists = torch.Tensor(self.input_dim, self.input_dim, num_steps)
        
        means = Variable(means)
        log_vars = Variable(log_vars)
        embeddings = Variable(embeddings)
        a_preds = Variable(a_preds)
        scaled_dists = Variable(scaled_dists)
        
        for i in range(num_steps):
            means[:, :, i] = self.mean(out[i, :, :])
            log_vars[:, :, i] = self.log_var(out[i, :, :])
            
            embeddings[:, :, i] = means[:, :, i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                  torch.sqrt(torch.exp(log_vars[:, :, i]))
                  
            dist = embeddings[:, :, i].unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
            dist = ((dist - dist.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
            a_preds[:, :, i] = 1 - F.tanh(torch.abs(self.a_pred(dist).view(\
                                      self.input_dim, self.input_dim)))
            scaled_dists[:, :, i] = torch.abs(self.scaled_dist(dist)).view(\
                                      self.input_dim, self.input_dim)
            
        return (means, log_vars, embeddings, a_preds, scaled_dists)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, scaled_dists, \
                                                      m=0.0, s=1.0, s4=1.0):
        """
            loss_function(ModelNNScale2, ndarray, ndarray, ndarray, ndarray, \
                                    ndarray, ndarray, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) Variational distri-
                        -bution variances
            a_preds: (input_dim, input_dim, num_steps) Output predictions from
                     the neural network
            scaled_dists: (input_dim, input_dim, time_steps) Scaled dist-
                               -tances between embeddings for influence
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
               vector is used
            s: Scale parameter for prior on w^(1)
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
            
        # Compute log p(z^(1))
        log_pz1 = -torch.sum((embeddings[:, :, 0] - m) ** 2) / (2 * s**2)
        
        # Compute prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((A * torch.log(1e-6 + a_preds) + (1-A) * torch.log(\
                           1e-6 + 1 - a_preds)).sum(dim=2), diagonal=1).sum()
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1))
        d2 = torch.exp(-scaled_dists)
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1] + Variable( \
                    torch.eye(n)))).sum(dim=1)).unsqueeze(1).expand(n, d)
            log_z_az -= torch.sum((embeddings[:, :, t] - means) ** 2) / (2 * s4**2)
        
        # Compute the entropy term
        ent = torch.sum(variances) / 2
        
        # Compute ELBO
        ELBO = log_pz1 + log_a_z + log_z_az + ent
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings):
        """
            gen_next_layer(ModelNNScale2, ndarray, ndarray) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            
            Returns:
                out: The predicted output probabilities for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-torch.abs(self.scaled_dist(dist)).view(\
                                      self.input_dim, self.input_dim))
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
        a_preds = 1 - F.tanh(torch.abs(self.a_pred(dist2).view(\
                                      self.input_dim, self.input_dim)))
        
        return a_preds









class ModelPoi(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, embedding_dim=8, num_layers=1):
        """
            __init__(ModelPoi, int, int, int) -> None
            Implements the model without splitting with a Bi-Directional LSTM
            and no neural network for computing output. Output matrix has posi-
            -tive integer weights. Poisson distribution is used at output.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            num_layers: Number of layers in the LSTM
        """
        super(ModelPoi, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for mean and variance
        self.mean = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelPoi) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def forward(self, A):
        """
            forward(ModelPoi, ndarray) -> (ndarray, ndarray, ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            
            Returns:
                means: (input_dim, embedding_dim, time_steps) Mean for each
                            node for each time step 
                log_vars: (input_dim, embedding_dim, time_steps) Log 
                          variance for each node for each time step
                embeddings: (input_dim, embedding_dim, time_steps) Embeddings
                            sampled from each node
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Get the means, log-variance, a_preds and embeddings
        means = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        log_vars = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        embeddings = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        
        means = Variable(means)
        log_vars = Variable(log_vars)
        embeddings = Variable(embeddings)
        
        for i in range(num_steps):
            means[:, :, i] = self.mean(out[i, :, :])
            log_vars[:, :, i] = self.log_var(out[i, :, :])
            
            embeddings[:, :, i] = means[:, :, i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                  torch.sqrt(torch.exp(log_vars[:, :, i]))
        
        return (means, log_vars, embeddings)
        

    
    
    def loss_function(self, A, embeddings, variances, m=0.0, s=1.0, \
                                                        s1=1.0, s2=1.0, s4=1.0):
        """
            loss_function(ModelPoi, ndarray, ndarray, ndarray, ndarray, float, \
                                                   float, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) Variational distri-
                        -bution variances
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
               vector is used
            s: Scale parameter for prior on w^(1)
            s1: Scale parameter for connections P(A|Z)
            s2: Scale parameter for the influence
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
            
        # Compute log p(z^(1))
        log_pz1 = -torch.sum((embeddings[:, :, 0] - m) ** 2) / (2 * s**2)
        
        # Compute distances between embeddings
        distances = Variable(torch.zeros(n, n, T))
        for t in range(T):
            dist = embeddings[:, :, t].unsqueeze(0).expand(n, n, d)
            dist = torch.sum((dist - dist.permute(1, 0, 2))**2, dim=2)
            distances[:, :, t] = dist
        
        # Compute prod_t P(A^(t) | z^(t))
        a_preds = torch.exp(-distances / (s1**2))
        log_a_z = torch.triu((-a_preds + A*torch.log(1e-6 + a_preds)).\
                         sum(dim=2), diagonal=1).sum()

        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1))
        d2 = torch.exp(-distances / (s2**2))
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1] + Variable( \
                    torch.eye(n)))).sum(dim=1)).unsqueeze(1).expand(n, d)
            log_z_az -= torch.sum((embeddings[:, :, t] - means) ** 2) / (2 * s4**2)
        
        # Compute the entropy term
        ent = torch.sum(variances) / 2
        
        # Compute ELBO
        ELBO = log_pz1 + log_a_z + log_z_az + ent
        
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, s1=1.0, s2=1.5):
        """
            gen_next_layer(ModelPoi, ndarray, ndarray, float, float) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            s1: Scale paramter for output predictions
            s2: Scale parameter for influence
            
            Returns:
                out: The predicted output mean for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-dist / s2 ** 2)
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)) ** 2).sum(dim=2)
        a_preds = torch.exp(-dist2 / (s1**2))
        
        return a_preds
        
        
        
        
        
        
        
        
        
class ModelNNPoi(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, embedding_dim=8, num_layers=1):
        """
            __init__(ModelNNPoi, int, int, int) -> None
            Implements the model without splitting with a Bi-Directional LSTM.
            Uses neural network for output predictions. Output matrix has posi-
            -tive integer weights. Uses a Poisson distribution.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            num_layers: Number of layers in the LSTM
        """
        super(ModelNNPoi, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for mean and variance
        self.mean = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        
        # Set up the output prediction layer
        self.a_pred = nn.Linear(self.embedding_dim, 1)
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelNNPoi) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def forward(self, A):
        """
            forward(ModelNNPoi, ndarray) -> (ndarray, ndarray, ndarray, \
                                                                ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            
            Returns:
                means: (input_dim, embedding_dim, time_steps) Mean for each
                            node for each time step 
                log_vars: (input_dim, embedding_dim, time_steps) Log 
                          variance for each node for each time step
                embeddings: (input_dim, embedding_dim, time_steps) Embeddings
                            sampled from each node
                a_preds: Predictions for the output as given by NN
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Get the means, log-variance, a_preds and embeddings
        means = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        log_vars = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        embeddings = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        a_preds = torch.Tensor(self.input_dim, self.input_dim, num_steps)
        
        means = Variable(means)
        log_vars = Variable(log_vars)
        embeddings = Variable(embeddings)
        a_preds = Variable(a_preds)
        
        for i in range(num_steps):
            means[:, :, i] = self.mean(out[i, :, :])
            log_vars[:, :, i] = self.log_var(out[i, :, :])
            
            embeddings[:, :, i] = means[:, :, i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                  torch.sqrt(torch.exp(log_vars[:, :, i]))
                  
            dist = embeddings[:, :, i].unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
            dist = (dist - dist.permute(1, 0, 2)).view(-1, \
                                            self.embedding_dim) ** 2
            a_preds[:, :, i] = self.a_pred(dist).view(\
                                      self.input_dim, self.input_dim) ** 2
        
        return (means, log_vars, embeddings, a_preds)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, m=0.0, s=1.0, \
                                                                s2=1.0, s4=1.0):
        """
            loss_function(ModelNN, ndarray, ndarray, ndarray, ndarray, ndarray,\
                                                   float, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) Variational distri-
                        -bution variances
            a_preds: (input_dim, input_dim, num_steps) Output predictions from
                     the neural network
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
               vector is used
            s: Scale parameter for prior on w^(1)
            s2: Scale parameter for the influence
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
            
        # Compute log p(z^(1))
        log_pz1 = -torch.sum((embeddings[:, :, 0] - m) ** 2) / (2 * s**2)
        
        # Compute distances between embeddings
        distances = Variable(torch.zeros(n, n, T))
        for t in range(T):
            dist = embeddings[:, :, t].unsqueeze(0).expand(n, n, d)
            dist = torch.sum((dist - dist.permute(1, 0, 2))**2, dim=2)
            distances[:, :, t] = dist
        
        # Compute prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((-a_preds + A*torch.log(1e-6 + a_preds)).\
                         sum(dim=2), diagonal=1).sum()

        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1))
        d2 = torch.exp(-distances / (s2**2))
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1] + Variable( \
                    torch.eye(n)))).sum(dim=1)).unsqueeze(1).expand(n, d)
            log_z_az -= torch.sum((embeddings[:, :, t] - means) ** 2) / (2 * s4**2)
        
        # Compute the entropy term
        ent = torch.sum(variances) / 2
        
        # Compute ELBO
        ELBO = log_pz1 + log_a_z + log_z_az + ent
        
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, s2=1.5):
        """
            gen_next_layer(ModelNNPoi, ndarray, ndarray, float) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            s2: Scale parameter for influence
            
            Returns:
                out: The predicted output mean for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-dist / s2 ** 2)
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = (dist2 - dist2.permute(1, 0, 2)).view(-1, self.embedding_dim)**2
        a_preds = self.a_pred(dist2).view(\
                                      self.input_dim, self.input_dim) ** 2
        
        return a_preds
        
        
        
        
        
        
        
        

class ModelNNScalePoi(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, embedding_dim=8, num_layers=1):
        """
            __init__(ModelNNScalePoi, int, int, int) -> None
            Implements the model without splitting with a Bi-Directional LSTM.
            Uses neural network for predicting scale parameter s1 in output
            prediction. Output matrix has positive integer weights. Uses
            Poisson distribution for output.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            num_layers: Number of layers in the LSTM
        """
        super(ModelNNScalePoi, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for mean and variance
        self.mean = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        
        # Set up the output prediction layer parameters
        self.s1 = nn.Parameter(torch.ones(1))
        self.b1 = nn.Parameter(torch.zeros(1))
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelNNScalePoi) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def compute_output(self, dist):
        """
            compute_output(ModelNNScalePoi, ndarray) -> ndarray
            
            Computes the output mean for Poisson distribution for each node pair
            
            dist: (input_dim, input_dim) Distances between emebddings of nodes
            
            Returns:
                output: The prediction of mean
        """
        return torch.exp(-(dist * (self.s1**2)) + self.b1**2)
        
    
    def forward(self, A):
        """
            forward(ModelNNScalePoi, ndarray) -> (ndarray, ndarray, ndarray, \
                                                                        ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            
            Returns:
                means: (input_dim, embedding_dim, time_steps) Mean for each
                            node for each time step 
                log_vars: (input_dim, embedding_dim, time_steps) Log 
                          variance for each node for each time step
                embeddings: (input_dim, embedding_dim, time_steps) Embeddings
                            sampled from each node
                a_preds: Predictions for the output as given by NN
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Get the means, log-variance, a_preds and embeddings
        means = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        log_vars = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        embeddings = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        a_preds = torch.Tensor(self.input_dim, self.input_dim, num_steps)
        
        means = Variable(means)
        log_vars = Variable(log_vars)
        embeddings = Variable(embeddings)
        a_preds = Variable(a_preds)
        
        for i in range(num_steps):
            means[:, :, i] = self.mean(out[i, :, :])
            log_vars[:, :, i] = self.log_var(out[i, :, :])
            
            embeddings[:, :, i] = means[:, :, i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                  torch.sqrt(torch.exp(log_vars[:, :, i]))
                  
            dist = embeddings[:, :, i].unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
            dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
            a_preds[:, :, i] = self.compute_output(dist)
            
        return (means, log_vars, embeddings, a_preds)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, m=0.0, s=1.0, \
                                                                s2=1.0, s4=1.0):
        """
            loss_function(ModelNNScalePoi, ndarray, ndarray, ndarray, ndarray, \
                                          ndarray, float, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) Variational distri-
                        -bution variances
            a_preds: (input_dim, input_dim, num_steps) Output predictions from
                     the neural network
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
               vector is used
            s: Scale parameter for prior on w^(1)
            s2: Scale parameter for the influence
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
            
        # Compute log p(z^(1))
        log_pz1 = -torch.sum((embeddings[:, :, 0] - m) ** 2) / (2 * s**2)
        
        # Compute distances between embeddings
        distances = Variable(torch.zeros(n, n, T))
        for t in range(T):
            dist = embeddings[:, :, t].unsqueeze(0).expand(n, n, d)
            dist = torch.sum((dist - dist.permute(1, 0, 2))**2, dim=2)
            distances[:, :, t] = dist
        
        # Compute prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((-a_preds + A*torch.log(1e-6 + a_preds)).\
                         sum(dim=2), diagonal=1).sum()
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1))
        d2 = torch.exp(-distances / (s2**2))
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1] + Variable( \
                    torch.eye(n)))).sum(dim=1)).unsqueeze(1).expand(n, d)
            log_z_az -= torch.sum((embeddings[:, :, t] - means) ** 2) / (2 * s4**2)
        
        # Compute the entropy term
        ent = torch.sum(variances) / 2
        
        # Compute ELBO
        ELBO = log_pz1 + log_a_z + log_z_az + ent
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, s2=1.5):
        """
            gen_next_layer(ModelNNScalePoi, ndarray, ndarray, float) -> \
                                                                        ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            s2: Scale parameter for influence
            
            Returns:
                out: The predicted output mean for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-dist / s2 ** 2)
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)) ** 2).sum(dim=2)
        a_preds = self.compute_output(dist2)
        
        return a_preds










class ModelNNScale2Poi(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, embedding_dim=8, num_layers=1):
        """
            __init__(ModelNNScale2Poi, int, int, int) -> None
            Implements the model without splitting with a Bi-Directional LSTM.
            Uses neural network for predicting scale parameter s1 in output
            prediction and scale parameter s2 for influence. Output matrix has 
            positive integer weights. Uses Poisson distribution for output.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            num_layers: Number of layers in the LSTM
        """
        super(ModelNNScale2Poi, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for mean and variance
        self.mean = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        
        # Set up the output prediction layer parameters
        self.s1 = nn.Parameter(torch.ones(1))
        self.b1 = nn.Parameter(torch.zeros(1))
        
        # Set up the influence parameters
        self.s2 = nn.Parameter(torch.ones(1))
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelNNScalePoi) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def compute_output(self, dist):
        """
            compute_output(ModelNNScale2Poi, ndarray) -> ndarray
            
            Computes the output mean for Poisson distribution for each node pair
            
            dist: (input_dim, input_dim) Distances between emebddings of nodes
            
            Returns:
                output: The prediction of mean
        """
        return torch.exp(-(dist * (self.s1**2)) + self.b1**2)
    
    
    def compute_scaled_dist(self, dist):
        """
            compute_scaled_dist(ModelNNScale2Poi, ndarray) -> ndarray
            
            Computes the scaled distances for influence
            
            dist: (input_dim, input_dim) Distances between emebddings of nodes
            
            Returns:
                scaled_dists: Scaled distances
        """
        return (dist * (self.s2**2))
    
        
    
    def forward(self, A):
        """
            forward(ModelNNScale2Poi, ndarray) -> (ndarray, ndarray, ndarray, \
                                                                        ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            
            Returns:
                means: (input_dim, embedding_dim, time_steps) Mean for each
                            node for each time step 
                log_vars: (input_dim, embedding_dim, time_steps) Log 
                          variance for each node for each time step
                embeddings: (input_dim, embedding_dim, time_steps) Embeddings
                            sampled from each node
                a_preds: Predictions for the output as given by NN
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Get the means, log-variance, a_preds and embeddings
        means = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        log_vars = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        embeddings = torch.Tensor(self.input_dim, self.embedding_dim, num_steps)
        a_preds = torch.Tensor(self.input_dim, self.input_dim, num_steps)
        scaled_dists = torch.Tensor(self.input_dim, self.input_dim, num_steps)
        
        means = Variable(means)
        log_vars = Variable(log_vars)
        embeddings = Variable(embeddings)
        a_preds = Variable(a_preds)
        scaled_dists = Variable(scaled_dists)
        
        for i in range(num_steps):
            means[:, :, i] = self.mean(out[i, :, :])
            log_vars[:, :, i] = self.log_var(out[i, :, :])
            
            embeddings[:, :, i] = means[:, :, i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                  torch.sqrt(torch.exp(log_vars[:, :, i]))
                  
            dist = embeddings[:, :, i].unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
            dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
            a_preds[:, :, i] = self.compute_output(dist)
            scaled_dists[:, :, i] = self.compute_scaled_dist(dist)
            
        return (means, log_vars, embeddings, a_preds, scaled_dists)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, scaled_dists, \
                                                        m=0.0, s=1.0, s4=1.0):
        """
            loss_function(ModelNNScale2Poi, ndarray, ndarray, ndarray, ndarray, \
                                        ndarray, ndarray, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) Variational distri-
                        -bution variances
            a_preds: (input_dim, input_dim, num_steps) Output predictions from
                     the neural network
            scaled_dists: (input_dim, input_dim, num_steps) Scaled distances from
                     the neural network for influence prediction
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
               vector is used
            s: Scale parameter for prior on w^(1)
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
            
        # Compute log p(z^(1))
        log_pz1 = -torch.sum((embeddings[:, :, 0] - m) ** 2) / (2 * s**2)
        
        # Compute prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((-a_preds + A*torch.log(1e-6 + a_preds)).\
                         sum(dim=2), diagonal=1).sum()
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1))
        d2 = torch.exp(-scaled_dists)
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1] + Variable( \
                    torch.eye(n)))).sum(dim=1)).unsqueeze(1).expand(n, d)
            log_z_az -= torch.sum((embeddings[:, :, t] - means) ** 2) / (2 * s4**2)
        
        # Compute the entropy term
        ent = torch.sum(variances) / 2
        
        # Compute ELBO
        ELBO = log_pz1 + log_a_z + log_z_az + ent
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings):
        """
            gen_next_layer(ModelNNScale2Poi, ndarray, ndarray) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            
            Returns:
                out: The predicted output probabilities for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-self.compute_scaled_dist(dist))
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)) ** 2).sum(dim=2)
        a_preds = self.compute_output(dist2)
        
        return a_preds








class ModelFull(nn.Module):
    def __init__(self, num_nodes, K, hidden_dim=32, embedding_dim=8, \
                 int_mu_dim=8, int_alpha_dim=8, num_layers=1):
        """
            __init__(ModelFull, int, int, int, int, int, int) -> None
            Implements the full LEM model with a Bi-Directional LSTM.
            Output matrix is binary.
            No neural network for output prediction and h prediction.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            int_mu_dim: Dimension of intermediate layer for mu (layer 1)
            int_alpha_dim: Dimension of intermediate layer for alpha
            num_layers: Number of layers in the LSTM
        """
        super(ModelFull, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.int_mu_dim = int_mu_dim
        self.int_alpha_dim = int_alpha_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for original mean and variance
        self.mean_orig = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)

        # Set up intermediate layers for alphas and layer one mus
        self.alpha_int = nn.Linear(self.hidden_dim*2, self.int_alpha_dim)
        self.mu_int = nn.Linear(self.hidden_dim*2, self.int_mu_dim)

        # Set up alphas and layer one mus calculation layers
        self.alpha_m = nn.Linear(self.input_dim * self.int_alpha_dim, \
                               self.embedding_dim)
        self.alpha_log_s = nn.Linear(self.input_dim * self.int_alpha_dim, \
                               self.embedding_dim)
        self.mu_m = nn.Linear(self.input_dim * self.int_mu_dim, \
                               K * self.embedding_dim)
        self.mu_log_s = nn.Linear(self.input_dim * self.int_mu_dim, \
                               K * self.embedding_dim)
        
        # Set up h calculation layer
        self.h = nn.Linear(self.hidden_dim*2, 1)
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelFull) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def forward(self, A, K, pi, s1=1.0, s3=1.5, s4=0.5):
        """
            forward(ModelFull, ndarray, ndarray, int, float, float, float) -> \
                (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, \
                 ndarray, ndarray, ndarray, ndarray, ndarray, ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            K: Number of communities in the first layer
            pi: (K,) Prior distribution over class
            s1: Scale parameter for computing output
            s3: Scale parameter for influence of a new community
            s4: Scale parameter for sampling embedding from mean
            
            Returns:
                mean_orig: (input_dim, embedding_dim, num_steps) Original \
                            means
                log_vars: (input_dim, embedding_dim, num_steps) Log \
                           variance for embeddings
                m_alpha: (embedding_dim, num_steps) Alpha mean for each \
                          step
                s_alpha: (embedding_dim, num_steps) Alpha log var for \
                          each time step
                alphas: (embedding_dim, num_steps) Sampled alphas
                h: (input_dim, num_steps) Change probability for each node
                m_mus: (K, embedding_dim) Mu mean
                s_mus: (K, embedding_dim) Mu log var
                mus: (K, embedding_dim) Sampled mus
                c: (input_dim, K) Class assignment probability for first
                    layer
                a_preds: (input_dim, input_dim, num_steps) Output preds
                mean_final: (input_dim, embedding_dim, num_steps) Mean for final
                            embeddings
                embeddings: (input_dim, embedding_dim, num_steps) Sampled \
                            embeddings
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Initialize the required variables
        mean_orig = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]
        log_vars = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]

        m_mus = Variable(torch.Tensor(K, self.embedding_dim))
        s_mus = Variable(torch.Tensor(K, self.embedding_dim))
        mus = Variable(torch.Tensor(K, self.embedding_dim))

        m_alpha = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]
        s_alpha = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]
        alphas = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]

        c = Variable(torch.Tensor(self.input_dim, K))

        h = [Variable(torch.Tensor(self.input_dim)) \
                       for _ in range(num_steps)]

        a_preds = [Variable(torch.Tensor(self.input_dim, self.input_dim)) \
                       for _ in range(num_steps)]
        
        mean_final = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]
        
        embeddings = [Variable(torch.Tensor(self.input_dim, \
                      self.embedding_dim)) for _ in range(num_steps)]

        # Get values for the variables
        for i in range(num_steps):
            # Get the original mean
            mean_orig[i] = self.mean_orig(out[i, :, :])

            # Get the variance for q(z)
            log_vars[i] = self.log_var(out[i, :, :])
            
            # Add the effect of splitting
            if i > 0:   # Take splitting into account from second layer onwards
                # Alpha and h would have been calculated at previous timestep
                alpha1 = alphas[i-1].unsqueeze(0).expand(self.input_dim, \
                                                    self.embedding_dim)
                curr_h = h[i-1].unsqueeze(1).expand(self.input_dim, \
                                                    self.embedding_dim)
                
                
                # Find emebddings taking into account mean_orig, alpha and h
                temp2 = alpha1 * curr_h + (1 - curr_h) * mean_orig[i]
                mean_final[i] = temp2
                embeddings[i] = temp2 + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                           torch.exp(log_vars[i]/2)
                  
            else:   # No splitting in first layer, so without split mean is used
                mean_final[i] = mean_orig[i]
                embeddings[i] = mean_orig[i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                           torch.exp(log_vars[i]/2)
            
            # Calculate alphas and h, should be done for all layers
            temp = F.tanh(self.alpha_int(out[i, :, :])).view(1, -1)
            m_alpha[i] = self.alpha_m(temp).view(-1)
            s_alpha[i] = self.alpha_log_s(temp).view(-1)
            alphas[i] = m_alpha[i] + \
                        Variable(torch.randn(self.embedding_dim)) * \
                                 torch.exp(s_alpha[i]/2)

            #alpha = alphas[i].expand(self.input_dim, self.embedding_dim)
            #h[i] = 1 - torch.tanh(((embeddings[i] - alpha) ** 2)\
            #                       .sum(dim=1) / s3**2)
            h[i] = F.sigmoid(self.h(out[i, :, :])).view(-1)
            
            # Compute output prediction for each layer
            dist = embeddings[i].unsqueeze(0).expand(self.input_dim, \
                                        self.input_dim, self.embedding_dim)
            dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
            a_preds[i] = 1 - torch.tanh(dist / s1**2)
            
            # Compute mus and c for first layer      
            if i == 0:
                temp3 = F.tanh(self.mu_int(out[i, :, :])).view(1, -1)
                m_mus = self.mu_m(temp3).view(K, self.embedding_dim)
                s_mus = self.mu_log_s(temp3).view(K, self.embedding_dim)
                mus = m_mus + \
                  Variable(torch.randn(K, self.embedding_dim)) * \
                           torch.exp(s_mus/2)

                dist1 = embeddings[i].unsqueeze(0).expand(K, self.input_dim, \
                                                        self.embedding_dim)
                dist1 = torch.sum((dist1 - mus.unsqueeze(0).expand(\
                            self.input_dim, K, self.embedding_dim).\
                                     permute(1, 0, 2))**2, dim=2).t()
                dist1 = torch.exp(-dist1 / (2 * s4**2))
                ditd1 = dist1 * pi.unsqueeze(0).expand(self.input_dim, K)
                c = (1e-6 + dist1) / (K * 1e-6 + dist1.sum(dim=1)).\
                                    unsqueeze(1).expand(self.input_dim, K)
                

        # Convert lists to tensors   
        mean_orig = torch.cat([x.unsqueeze(2) for x in mean_orig], dim=2)
        log_vars = torch.cat([x.unsqueeze(2) for x in log_vars], dim=2)
        m_alpha = torch.cat([x.unsqueeze(1) for x in m_alpha], dim=1)
        s_alpha = torch.cat([x.unsqueeze(1) for x in s_alpha], dim=1)
        alphas = torch.cat([x.unsqueeze(1) for x in alphas], dim=1)
        h = torch.cat([x.unsqueeze(1) for x in h], dim=1)
        a_preds = torch.cat([x.unsqueeze(2) for x in a_preds], dim=2)
        mean_final = torch.cat([x.unsqueeze(2) for x in mean_final], dim=2)
        embeddings = torch.cat([x.unsqueeze(2) for x in embeddings], dim=2)
        
        return (mean_orig, log_vars, m_alpha, s_alpha, alphas, h, m_mus, \
                s_mus, mus, c, a_preds, mean_final, embeddings)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, alphas, mus, \
                        s_alpha, s_mus, c, h, pi, m=0.0, s=1.0, s1=1.0, \
                        s2=1.5, s3=1.5, s4=0.5):
        """
            loss_function(Model, ndarray, ndarray, ndarray, ndarray, \
                          ndarray, ndarray, ndarray, ndarray, ndarray, \
                          ndarray, ndarray, float, float, float, float, \
                          float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) variances of em-
                        -beddings
            a_preds: (input_dim, input_dim, num_steps) Output probability
            alphas: (embedding_dim, num_steps) Sampled alphas
                    m_mus: (K, embedding_dim) Mu mean
            mus: (K, embedding_dim) Sampled mus
            s_alpha: (embedding_dim, num_steps) Alpha log var for \
                              each time step
            s_mus: (K, embedding_dim) Mu log var
            c: (input_dim, K) Class assignment probability for first
                        layer
            h: (input_dim, num_steps) Change probability for each node
            pi: (K,) Class membership prior for first layer
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
                   vector is used
            s: Scale parameter for prior on w^(1)
            s1: Scale parameter for connections P(A|Z)
            s2: Scale parameter for the influence
            s3: Scale parameter for influence of a new community
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
        K = pi.size()[0]

        # Compute log p(mu)
        log_pmu = -torch.sum((mus - Variable(torch.zeros(K, d))) ** 2) / (2*s**2)
                
        # Compute log p(alphas)
        log_palpha = -torch.sum((alphas - Variable(torch.zeros(d, T)))\
                      ** 2) / (2 * s**2)
        
        # Compute log P(c)
        log_pc = torch.sum(c * torch.log(1e-6 + pi.unsqueeze(0).expand(n, K)))
        
        # Compute distances between layer 1 embeddings and mus
        dist_l1 = embeddings[:, :, 0].unsqueeze(0).expand(K, n, d)
        dist_l1 = torch.sum((dist_l1 - mus.unsqueeze(0).expand(n, K, d).\
                             permute(1, 0, 2))**2, dim=2).t()
        dist_l1 = -dist_l1 / (2 * s4**2)
        log_pz1 = torch.sum(c * dist_l1)
        
        # Compute distances between embeddings and alphas for all layers
        distances = Variable(torch.zeros(n, T))
        for t in range(T):
            dist = torch.sum((embeddings[:, :, t] - \
                              alphas[:, t].unsqueeze(0).expand(n, d))**2, \
                              dim=1)
            distances[:, t] = dist / s3**2
        
        # Compute log p(h) for all layers
        log_ph = 1 - torch.tanh(distances)
        log_ph = h * torch.log(1e-6 + log_ph) + \
                 (1-h) * torch.log(1e-6 + 1-log_ph)
        log_ph = log_ph.sum()
        
        # Compute distances between embeddings
        distances1 = Variable(torch.zeros(n, n, T))
        for t in range(T):
            dist = embeddings[:, :, t].unsqueeze(0).expand(n, n, d)
            dist = torch.sum((dist - dist.permute(1, 0, 2))**2, dim=2)
            distances1[:, :, t] = dist
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1), h(t), alpha(t))
        d2 = torch.exp(-distances1 / (s2**2))
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1]+Variable(torch.eye(n))))\
                    .sum(dim=1)).unsqueeze(1).expand(n, d)
            
            log_z_az -= (((1-h[:, t-1]) * ((embeddings[:, :, t] - \
                                        means) ** 2).sum(dim=1)) / (2 * s4**2) + \
                        (h[:, t-1] * ((embeddings[:, :, t] - \
                                        alphas[:, t-1].unsqueeze(0).expand(n, d)) \
                                        ** 2).sum(dim=1)) / (2 * s4**2)).sum()
        
        # Compute  prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((A * torch.log(1e-6 + a_preds) + (1-A) * \
                              torch.log(1e-6 + 1 - a_preds)).\
                             sum(dim=2), diagonal=1).sum()
        
        # Compute the entropy term for z's
        ent_z = torch.sum(variances) / 2
        
        # Compute the entropy term for alpha's
        ent_alpha = torch.sum(s_alpha) / 2
        
        # Compute the entropy term for mus's
        ent_mu = torch.sum(s_mus) / 2        
        
        # Compute the entropy term for c
        ent_c = -(c * torch.log(1e-6 + c)).sum()
        
        # Compute entropy for hs
        ent_h = -(h*torch.log(1e-6 + h) + (1-h)*torch.log(1e-6 + 1 - h)).sum()
        
        # Compute ELBO
        ELBO = log_pmu + log_a_z + log_palpha + log_ph + log_z_az + log_pc + \
               log_pz1 + ent_z + ent_alpha + ent_mu + ent_c + ent_h
        
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, h, alpha, s1=1.0, s2=1.5):
        """
            gen_next_layer(Model, ndarray, ndarray, ndarray, ndarray, float, \
                                                    float) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            h: (input_dim,) Change probability for layer t
            alpha: (embedding_dim,) New community center for layer t
            s1: Scale paramter for output predictions
            s2: Scale parameter for influence
            
            Returns:
                out: The predicted output probabilities for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-dist / s2 ** 2)
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        #h = h.unsqueeze(1).expand(n, d)
        #alpha = alpha.unsqueeze(0).expand(n, d)
        #Z_next = (1-h) * Z_next + h * alpha
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)) ** 2).sum(dim=2)
        a_preds = 1 - torch.tanh(dist2 / (s1**2))
        
        return a_preds
        
        
        






class ModelFullScaled(nn.Module):
    def __init__(self, num_nodes, K, hidden_dim=32, embedding_dim=8, \
                 int_mu_dim=8, int_alpha_dim=8, num_layers=1, \
                 use_bias_s1=False, use_bias_s2=False):
        """
            __init__(ModelFullScaled, int, int, int, int, int, int, bool, \
                                                    bool) -> None
            Implements the full LEM model with a Bi-Directional LSTM.
            Output matrix is binary.
            No neural network for output prediction and h prediction.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            int_mu_dim: Dimension of intermediate layer for mu (layer 1)
            int_alpha_dim: Dimension of intermediate layer for alpha
            num_layers: Number of layers in the LSTM
            use_bias_s1: Whether to use bias in output scale calculation
            use_bias_s2: Whether to use bias in influence scale calculation
        """
        super(ModelFullScaled, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.int_mu_dim = int_mu_dim
        self.int_alpha_dim = int_alpha_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for original mean and variance
        self.mean_orig = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)

        # Set up intermediate layers for alphas and layer one mus
        self.alpha_int = nn.Linear(self.hidden_dim*2, self.int_alpha_dim)
        self.mu_int = nn.Linear(self.hidden_dim*2, self.int_mu_dim)

        # Set up alphas and layer one mus calculation layers
        self.alpha_m = nn.Linear(self.input_dim * self.int_alpha_dim, \
                               self.embedding_dim)
        self.alpha_log_s = nn.Linear(self.input_dim * self.int_alpha_dim, \
                               self.embedding_dim)
        self.mu_m = nn.Linear(self.input_dim * self.int_mu_dim, \
                               K * self.embedding_dim)
        self.mu_log_s = nn.Linear(self.input_dim * self.int_mu_dim, \
                               K * self.embedding_dim)
        
        # Set up h calculation layer
        self.h = nn.Linear(self.hidden_dim*2, 1)
        
        # Set up output prediction layer
        self.a_pred = nn.Linear(1, 1, bias=use_bias_s1)
        
        # Set up scaled distance layer for influence
        self.sc_dist_inf = nn.Linear(1, 1, bias=use_bias_s2)
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelFullScaled2) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def forward(self, A, K, pi, s3=1.0, s4=0.5):
        """
            forward(ModelFullScaled2, ndarray, ndarray, int, float, float) -> \
                (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, \
                 ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, \
                 ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            K: Number of communities in the first layer
            pi: (K,) Prior distribution over class
            s4: Scale parameter for sampling embedding from mean
            
            Returns:
                mean_orig: (input_dim, embedding_dim, num_steps) Original \
                            means
                log_vars: (input_dim, embedding_dim, num_steps) Log \
                           variance for embeddings
                m_alpha: (embedding_dim, num_steps) Alpha mean for each \
                          step
                s_alpha: (embedding_dim, num_steps) Alpha log var for \
                          each time step
                alphas: (embedding_dim, num_steps) Sampled alphas
                h: (input_dim, num_steps) Change probability for each node
                m_mus: (K, embedding_dim) Mu mean
                s_mus: (K, embedding_dim) Mu log var
                mus: (K, embedding_dim) Sampled mus
                c: (input_dim, K) Class assignment probability for first
                    layer
                a_preds: (input_dim, input_dim, num_steps) Output preds
                sc_dist_inf: (input_dim, input_dim, num_steps) Scaled distance
                             for influence prediction
                mean_final: (input_dim, embedding_dim, num_steps) Mean for final
                            embeddings
                embeddings: (input_dim, embedding_dim, num_steps) Sampled \
                            embeddings
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Initialize the required variables
        mean_orig = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]
        log_vars = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]

        m_mus = Variable(torch.Tensor(K, self.embedding_dim))
        s_mus = Variable(torch.Tensor(K, self.embedding_dim))
        mus = Variable(torch.Tensor(K, self.embedding_dim))

        m_alpha = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]
        s_alpha = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]
        alphas = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]

        c = Variable(torch.Tensor(self.input_dim, K))

        h = [Variable(torch.Tensor(self.input_dim)) \
                       for _ in range(num_steps)]

        a_preds = [Variable(torch.Tensor(self.input_dim, self.input_dim)) \
                       for _ in range(num_steps)]
                       
        sc_dist_inf = [Variable(torch.Tensor(self.input_dim, self.input_dim)) \
                       for _ in range(num_steps)]
        
        mean_final = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]
        
        embeddings = [Variable(torch.Tensor(self.input_dim, \
                      self.embedding_dim)) for _ in range(num_steps)]

        # Get values for the variables
        for i in range(num_steps):
            # Get the original mean
            mean_orig[i] = self.mean_orig(out[i, :, :])

            # Get the variance for q(z)
            log_vars[i] = self.log_var(out[i, :, :])
            
            # Add the effect of splitting
            if i > 0:   # Take splitting into account from second layer onwards
                # Alpha and h would have been calculated at previous timestep
                alpha1 = alphas[i-1].unsqueeze(0).expand(self.input_dim, \
                                                    self.embedding_dim)
                curr_h = h[i-1].unsqueeze(1).expand(self.input_dim, \
                                                    self.embedding_dim)
                
                
                # Find emebddings taking into account mean_orig, alpha and h
                temp2 = alpha1 * curr_h + (1 - curr_h) * mean_orig[i]
                mean_final[i] = temp2
                embeddings[i] = temp2 + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                           torch.exp(log_vars[i]/2)
                  
            else:   # No splitting in first layer, so without split mean is used
                mean_final[i] = mean_orig[i]
                embeddings[i] = mean_orig[i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                           torch.exp(log_vars[i]/2)
            
            # Calculate alphas and h, should be done for all layers
            temp = F.tanh(self.alpha_int(out[i, :, :])).view(1, -1)
            m_alpha[i] = self.alpha_m(temp).view(-1)
            s_alpha[i] = self.alpha_log_s(temp).view(-1)
            alphas[i] = m_alpha[i] + \
                        Variable(torch.randn(self.embedding_dim)) * \
                                 torch.exp(s_alpha[i]/2)

            h[i] = F.sigmoid(self.h(out[i, :, :])).view(-1)
            
            # Compute output prediction for each layer
            dist = embeddings[i].unsqueeze(0).expand(self.input_dim, \
                                        self.input_dim, self.embedding_dim)
            dist = ((dist - dist.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
            a_preds[i] = 1 - torch.tanh(torch.abs(self.a_pred(dist).view(\
                                      self.input_dim, self.input_dim)))
            sc_dist_inf[i] = torch.abs(self.sc_dist_inf(dist)).view(\
                                      self.input_dim, self.input_dim)
            
            # Compute mus and c for first layer      
            if i == 0:
                temp3 = F.tanh(self.mu_int(out[i, :, :])).view(1, -1)
                m_mus = self.mu_m(temp3).view(K, self.embedding_dim)
                s_mus = self.mu_log_s(temp3).view(K, self.embedding_dim)
                mus = m_mus + \
                  Variable(torch.randn(K, self.embedding_dim)) * \
                           torch.exp(s_mus/2)

                dist1 = embeddings[i].unsqueeze(0).expand(K, self.input_dim, \
                                                        self.embedding_dim)
                dist1 = torch.sum((dist1 - mus.unsqueeze(0).expand(\
                            self.input_dim, K, self.embedding_dim).\
                                     permute(1, 0, 2))**2, dim=2).t()
                dist1 = torch.exp(-dist1 / (2 * s4**2))
                ditd1 = dist1 * pi.unsqueeze(0).expand(self.input_dim, K)
                c = (1e-6 + dist1) / (K * 1e-6 + dist1.sum(dim=1)).\
                                    unsqueeze(1).expand(self.input_dim, K)
                

        # Convert lists to tensors   
        mean_orig = torch.cat([x.unsqueeze(2) for x in mean_orig], dim=2)
        log_vars = torch.cat([x.unsqueeze(2) for x in log_vars], dim=2)
        m_alpha = torch.cat([x.unsqueeze(1) for x in m_alpha], dim=1)
        s_alpha = torch.cat([x.unsqueeze(1) for x in s_alpha], dim=1)
        alphas = torch.cat([x.unsqueeze(1) for x in alphas], dim=1)
        h = torch.cat([x.unsqueeze(1) for x in h], dim=1)
        a_preds = torch.cat([x.unsqueeze(2) for x in a_preds], dim=2)
        sc_dist_inf = torch.cat([x.unsqueeze(2) for x in sc_dist_inf], dim=2)
        mean_final = torch.cat([x.unsqueeze(2) for x in mean_final], dim=2)
        embeddings = torch.cat([x.unsqueeze(2) for x in embeddings], dim=2)
        
        return (mean_orig, log_vars, m_alpha, s_alpha, alphas, h, m_mus, \
                s_mus, mus, c, a_preds, sc_dist_inf, mean_final, embeddings)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, sc_dist_inf, \
                      alphas, mus, s_alpha, s_mus, c, h, pi, \
                      m=0.0, s=1.0, s3=1.0, s4=0.5):
        """
            loss_function(ModelFullScaled, ndarray, ndarray, ndarray, \
                          ndarray, ndarray, ndarray, ndarray, ndarray, \
                          ndarray, ndarray, ndarray, ndarray, float, \
                          float, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) variances of em-
                        -beddings
            a_preds: (input_dim, input_dim, num_steps) Output probability
            sc_dist_inf: (input_dim, input_dim, num_steps) Scaled distance
                         for influence prediction
            sc_dist_new: (input_dim, num_steps) Scaled distance
                         for new community influence prediction
            alphas: (embedding_dim, num_steps) Sampled alphas
                    m_mus: (K, embedding_dim) Mu mean
            mus: (K, embedding_dim) Sampled mus
            s_alpha: (embedding_dim, num_steps) Alpha log var for \
                              each time step
            s_mus: (K, embedding_dim) Mu log var
            c: (input_dim, K) Class assignment probability for first
                        layer
            h: (input_dim, num_steps) Change probability for each node
            pi: (K,) Class membership prior for first layer
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
                   vector is used
            s: Scale parameter for prior on w^(1)
            s4: Scale parameter for influence of new community
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
        K = pi.size()[0]

        # Compute log p(mu)
        log_pmu = -torch.sum((mus - Variable(torch.zeros(K, d))) ** 2) / (2*s**2)
                
        # Compute log p(alphas)
        log_palpha = -torch.sum((alphas - Variable(torch.zeros(d, T)))\
                      ** 2) / (2 * s**2)
        
        # Compute log P(c)
        log_pc = torch.sum(c * torch.log(1e-6 + pi.unsqueeze(0).expand(n, K)))
        
        # Compute distances between layer 1 embeddings and mus
        dist_l1 = embeddings[:, :, 0].unsqueeze(0).expand(K, n, d)
        dist_l1 = torch.sum((dist_l1 - mus.unsqueeze(0).expand(n, K, d).\
                             permute(1, 0, 2))**2, dim=2).t()
        dist_l1 = -dist_l1 / (2 * s4**2)
        log_pz1 = torch.sum(c * dist_l1)
        
        # Compute distances between embeddings and alphas for all layers
        distances = Variable(torch.zeros(n, T))
        for t in range(T):
            dist = torch.sum((embeddings[:, :, t] - \
                              alphas[:, t].unsqueeze(0).expand(n, d))**2, \
                              dim=1)
            distances[:, t] = dist / s3**2
        
        # Compute log p(h) for all layers
        log_ph = 1 - torch.tanh(distances)
        log_ph = h * torch.log(1e-6 + log_ph) + \
                 (1-h) * torch.log(1e-6 + 1-log_ph)
        log_ph = log_ph.sum()
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1), h(t), alpha(t))
        d2 = torch.exp(-sc_dist_inf)
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1]+Variable(torch.eye(n))))\
                    .sum(dim=1)).unsqueeze(1).expand(n, d)
            
            log_z_az -= (((1-h[:, t-1]) * ((embeddings[:, :, t] - \
                                        means) ** 2).sum(dim=1)) / (2 * s4**2) + \
                        (h[:, t-1] * ((embeddings[:, :, t] - \
                                        alphas[:, t-1].unsqueeze(0).expand(n, d)) \
                                        ** 2).sum(dim=1)) / (2 * s4**2)).sum()
        
        # Compute  prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((A * torch.log(1e-6 + a_preds) + (1-A) * \
                              torch.log(1e-6 + 1 - a_preds)).\
                             sum(dim=2), diagonal=1).sum()
        
        # Compute the entropy term for z's
        ent_z = torch.sum(variances) / 2
        
        # Compute the entropy term for alpha's
        ent_alpha = torch.sum(s_alpha) / 2
        
        # Compute the entropy term for mus's
        ent_mu = torch.sum(s_mus) / 2        
        
        # Compute the entropy term for c
        ent_c = -(c * torch.log(1e-6 + c)).sum()
        
        # Compute entropy for hs
        ent_h = -(h*torch.log(1e-6 + h) + (1-h)*torch.log(1e-6 + 1 - h)).sum()
        
        # Compute ELBO
        ELBO = log_pmu + log_a_z + log_palpha + log_ph + log_z_az + log_pc + \
               log_pz1 + ent_z + ent_alpha + ent_mu + ent_c + ent_h
        
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, h, alpha):
        """
            gen_next_layer(Model, ndarray, ndarray, ndarray, ndarray) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            h: (input_dim,) Change probability for layer t
            alpha: (embedding_dim,) New community center for layer t
            
            Returns:
                out: The predicted output probabilities for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-torch.abs(self.sc_dist_inf(dist)).view(\
                                      self.input_dim, self.input_dim))
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        #h = h.unsqueeze(1).expand(n, d)
        #alpha = alpha.unsqueeze(0).expand(n, d)
        #Z_next = (1-h) * Z_next + h * alpha
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
        a_preds = 1 - F.tanh(torch.abs(self.a_pred(dist2).view(\
                                      self.input_dim, self.input_dim)))
        
        return a_preds
        
        
        
        
        
        
        
        
class ModelPoiFullScaled(nn.Module):
    def __init__(self, num_nodes, K, hidden_dim=32, embedding_dim=8, \
                 int_mu_dim=8, int_alpha_dim=8, num_layers=1, \
                 use_bias_s1=False, use_bias_s2=False):
        """
            __init__(ModelPoiFullScaled, int, int, int, int, int, int, bool, \
                                                    bool) -> None
            Implements the full LEM model with a Bi-Directional LSTM.
            Output matrix is weighted.
            No neural network for output prediction and h prediction.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            int_mu_dim: Dimension of intermediate layer for mu (layer 1)
            int_alpha_dim: Dimension of intermediate layer for alpha
            num_layers: Number of layers in the LSTM
            use_bias_s1: Whether to use bias in output scale calculation
            use_bias_s2: Whether to use bias in influence scale calculation
        """
        super(ModelPoiFullScaled, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.int_mu_dim = int_mu_dim
        self.int_alpha_dim = int_alpha_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for original mean and variance
        self.mean_orig = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)

        # Set up intermediate layers for alphas and layer one mus
        self.alpha_int = nn.Linear(self.hidden_dim*2, self.int_alpha_dim)
        self.mu_int = nn.Linear(self.hidden_dim*2, self.int_mu_dim)

        # Set up alphas and layer one mus calculation layers
        self.alpha_m = nn.Linear(self.input_dim * self.int_alpha_dim, \
                               self.embedding_dim)
        self.alpha_log_s = nn.Linear(self.input_dim * self.int_alpha_dim, \
                               self.embedding_dim)
        self.mu_m = nn.Linear(self.input_dim * self.int_mu_dim, \
                               K * self.embedding_dim)
        self.mu_log_s = nn.Linear(self.input_dim * self.int_mu_dim, \
                               K * self.embedding_dim)
        
        # Set up h calculation layer
        self.h = nn.Linear(self.hidden_dim*2, 1)
        
        # Set up the output prediction layer parameters
        self.s1 = nn.Parameter(torch.ones(1))
        self.b1 = nn.Parameter(torch.zeros(1))
        
        # Set up scaled distance layer for influence
        self.s2 = nn.Parameter(torch.ones(1))
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelPoiFullScaled) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def compute_scaled_dist(self, dist):
        """
            compute_scaled_dist(ModelNNScale2Poi, ndarray) -> ndarray
            
            Computes the scaled distances for influence
            
            dist: (input_dim, input_dim) Distances between emebddings of nodes
            
            Returns:
                scaled_dists: Scaled distances
        """
        return (dist * (self.s2**2))
    
    
    def compute_output(self, dist):
        """
            compute_output(ModelPoiFullScaled, ndarray) -> ndarray
            
            Computes the output mean for Poisson distribution for each node pair
            
            dist: (input_dim, input_dim) Distances between emebddings of nodes
            
            Returns:
                output: The prediction of mean
        """
        return torch.exp(-(dist * (self.s1**2)) + self.b1**2)
    
    
    
    def forward(self, A, K, pi, s3=1.0, s4=0.5):
        """
            forward(ModelPoiFullScaled, ndarray, ndarray, int, float, float) -> \
                (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, \
                 ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, \
                 ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            K: Number of communities in the first layer
            pi: (K,) Prior distribution over class
            s4: Scale parameter for sampling embedding from mean
            
            Returns:
                mean_orig: (input_dim, embedding_dim, num_steps) Original \
                            means
                log_vars: (input_dim, embedding_dim, num_steps) Log \
                           variance for embeddings
                m_alpha: (embedding_dim, num_steps) Alpha mean for each \
                          step
                s_alpha: (embedding_dim, num_steps) Alpha log var for \
                          each time step
                alphas: (embedding_dim, num_steps) Sampled alphas
                h: (input_dim, num_steps) Change probability for each node
                m_mus: (K, embedding_dim) Mu mean
                s_mus: (K, embedding_dim) Mu log var
                mus: (K, embedding_dim) Sampled mus
                c: (input_dim, K) Class assignment probability for first
                    layer
                a_preds: (input_dim, input_dim, num_steps) Output preds
                sc_dist_inf: (input_dim, input_dim, num_steps) Scaled distance
                             for influence prediction
                mean_final: (input_dim, embedding_dim, num_steps) Mean for final
                            embeddings
                embeddings: (input_dim, embedding_dim, num_steps) Sampled \
                            embeddings
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Initialize the required variables
        mean_orig = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]
        log_vars = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]

        m_mus = Variable(torch.Tensor(K, self.embedding_dim))
        s_mus = Variable(torch.Tensor(K, self.embedding_dim))
        mus = Variable(torch.Tensor(K, self.embedding_dim))

        m_alpha = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]
        s_alpha = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]
        alphas = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]

        c = Variable(torch.Tensor(self.input_dim, K))

        h = [Variable(torch.Tensor(self.input_dim)) \
                       for _ in range(num_steps)]

        a_preds = [Variable(torch.Tensor(self.input_dim, self.input_dim)) \
                       for _ in range(num_steps)]
                       
        sc_dist_inf = [Variable(torch.Tensor(self.input_dim, self.input_dim)) \
                       for _ in range(num_steps)]
        
        mean_final = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]
        
        embeddings = [Variable(torch.Tensor(self.input_dim, \
                      self.embedding_dim)) for _ in range(num_steps)]

        # Get values for the variables
        for i in range(num_steps):
            # Get the original mean
            mean_orig[i] = self.mean_orig(out[i, :, :])

            # Get the variance for q(z)
            log_vars[i] = self.log_var(out[i, :, :])
            
            # Add the effect of splitting
            if i > 0:   # Take splitting into account from second layer onwards
                # Alpha and h would have been calculated at previous timestep
                alpha1 = alphas[i-1].unsqueeze(0).expand(self.input_dim, \
                                                    self.embedding_dim)
                curr_h = h[i-1].unsqueeze(1).expand(self.input_dim, \
                                                    self.embedding_dim)
                
                
                # Find emebddings taking into account mean_orig, alpha and h
                temp2 = alpha1 * curr_h + (1 - curr_h) * mean_orig[i]
                mean_final[i] = temp2
                embeddings[i] = temp2 + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                           torch.exp(log_vars[i]/2)
                  
            else:   # No splitting in first layer, so without split mean is used
                mean_final[i] = mean_orig[i]
                embeddings[i] = mean_orig[i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                           torch.exp(log_vars[i]/2)
            
            # Calculate alphas and h, should be done for all layers
            temp = F.tanh(self.alpha_int(out[i, :, :])).view(1, -1)
            m_alpha[i] = self.alpha_m(temp).view(-1)
            s_alpha[i] = self.alpha_log_s(temp).view(-1)
            alphas[i] = m_alpha[i] + \
                        Variable(torch.randn(self.embedding_dim)) * \
                                 torch.exp(s_alpha[i]/2)

            h[i] = F.sigmoid(self.h(out[i, :, :])).view(-1)
            
            # Compute output prediction for each layer
            dist = embeddings[i].unsqueeze(0).expand(self.input_dim, \
                                        self.input_dim, self.embedding_dim)
            dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
            a_preds[i] = self.compute_output(dist)
            sc_dist_inf[i] = self.compute_scaled_dist(dist)
            
            # Compute mus and c for first layer      
            if i == 0:
                temp3 = F.tanh(self.mu_int(out[i, :, :])).view(1, -1)
                m_mus = self.mu_m(temp3).view(K, self.embedding_dim)
                s_mus = self.mu_log_s(temp3).view(K, self.embedding_dim)
                mus = m_mus + \
                  Variable(torch.randn(K, self.embedding_dim)) * \
                           torch.exp(s_mus/2)

                dist1 = embeddings[i].unsqueeze(0).expand(K, self.input_dim, \
                                                        self.embedding_dim)
                dist1 = torch.sum((dist1 - mus.unsqueeze(0).expand(\
                            self.input_dim, K, self.embedding_dim).\
                                     permute(1, 0, 2))**2, dim=2).t()
                dist1 = torch.exp(-dist1 / (2 * s4**2))
                ditd1 = dist1 * pi.unsqueeze(0).expand(self.input_dim, K)
                c = (1e-6 + dist1) / (K * 1e-6 + dist1.sum(dim=1)).\
                                    unsqueeze(1).expand(self.input_dim, K)
                

        # Convert lists to tensors   
        mean_orig = torch.cat([x.unsqueeze(2) for x in mean_orig], dim=2)
        log_vars = torch.cat([x.unsqueeze(2) for x in log_vars], dim=2)
        m_alpha = torch.cat([x.unsqueeze(1) for x in m_alpha], dim=1)
        s_alpha = torch.cat([x.unsqueeze(1) for x in s_alpha], dim=1)
        alphas = torch.cat([x.unsqueeze(1) for x in alphas], dim=1)
        h = torch.cat([x.unsqueeze(1) for x in h], dim=1)
        a_preds = torch.cat([x.unsqueeze(2) for x in a_preds], dim=2)
        sc_dist_inf = torch.cat([x.unsqueeze(2) for x in sc_dist_inf], dim=2)
        mean_final = torch.cat([x.unsqueeze(2) for x in mean_final], dim=2)
        embeddings = torch.cat([x.unsqueeze(2) for x in embeddings], dim=2)
        
        return (mean_orig, log_vars, m_alpha, s_alpha, alphas, h, m_mus, \
                s_mus, mus, c, a_preds, sc_dist_inf, mean_final, embeddings)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, sc_dist_inf, \
                      alphas, mus, s_alpha, s_mus, c, h, pi, \
                      m=0.0, s=1.0, s3=1.0, s4=0.5):
        """
            loss_function(ModelPoiFullScaled, ndarray, ndarray, ndarray, \
                          ndarray, ndarray, ndarray, ndarray, ndarray, \
                          ndarray, ndarray, ndarray, ndarray, float, \
                          float, float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) variances of em-
                        -beddings
            a_preds: (input_dim, input_dim, num_steps) Output probability
            sc_dist_inf: (input_dim, input_dim, num_steps) Scaled distance
                         for influence prediction
            sc_dist_new: (input_dim, num_steps) Scaled distance
                         for new community influence prediction
            alphas: (embedding_dim, num_steps) Sampled alphas
                    m_mus: (K, embedding_dim) Mu mean
            mus: (K, embedding_dim) Sampled mus
            s_alpha: (embedding_dim, num_steps) Alpha log var for \
                              each time step
            s_mus: (K, embedding_dim) Mu log var
            c: (input_dim, K) Class assignment probability for first
                        layer
            h: (input_dim, num_steps) Change probability for each node
            pi: (K,) Class membership prior for first layer
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
                   vector is used
            s: Scale parameter for prior on w^(1)
            s4: Scale parameter for influence of new community
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
        K = pi.size()[0]

        # Compute log p(mu)
        log_pmu = -torch.sum((mus - Variable(torch.zeros(K, d))) ** 2) / (2*s**2)
                
        # Compute log p(alphas)
        log_palpha = -torch.sum((alphas - Variable(torch.zeros(d, T)))\
                      ** 2) / (2 * s**2)
        
        # Compute log P(c)
        log_pc = torch.sum(c * torch.log(1e-6 + pi.unsqueeze(0).expand(n, K)))
        
        # Compute distances between layer 1 embeddings and mus
        dist_l1 = embeddings[:, :, 0].unsqueeze(0).expand(K, n, d)
        dist_l1 = torch.sum((dist_l1 - mus.unsqueeze(0).expand(n, K, d).\
                             permute(1, 0, 2))**2, dim=2).t()
        dist_l1 = -dist_l1 / (2 * s4**2)
        log_pz1 = torch.sum(c * dist_l1)
        
        # Compute distances between embeddings and alphas for all layers
        distances = Variable(torch.zeros(n, T))
        for t in range(T):
            dist = torch.sum((embeddings[:, :, t] - \
                              alphas[:, t].unsqueeze(0).expand(n, d))**2, \
                              dim=1)
            distances[:, t] = dist / s3**2
        
        # Compute log p(h) for all layers
        log_ph = 1 - torch.tanh(distances)
        log_ph = h * torch.log(1e-6 + log_ph) + \
                 (1-h) * torch.log(1e-6 + 1-log_ph)
        log_ph = log_ph.sum()
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1), h(t), alpha(t))
        d2 = torch.exp(-sc_dist_inf)
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1]+Variable(torch.eye(n))))\
                    .sum(dim=1)).unsqueeze(1).expand(n, d)
            
            log_z_az -= (((1-h[:, t-1]) * ((embeddings[:, :, t] - \
                                        means) ** 2).sum(dim=1)) / (2 * s4**2) + \
                        (h[:, t-1] * ((embeddings[:, :, t] - \
                                        alphas[:, t-1].unsqueeze(0).expand(n, d)) \
                                        ** 2).sum(dim=1)) / (2 * s4**2)).sum()
        
        # Compute  prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((-a_preds + A*torch.log(1e-6 + a_preds)).\
                         sum(dim=2), diagonal=1).sum()
        
        # Compute the entropy term for z's
        ent_z = torch.sum(variances) / 2
        
        # Compute the entropy term for alpha's
        ent_alpha = torch.sum(s_alpha) / 2
        
        # Compute the entropy term for mus's
        ent_mu = torch.sum(s_mus) / 2        
        
        # Compute the entropy term for c
        ent_c = -(c * torch.log(1e-6 + c)).sum()
        
        # Compute entropy for hs
        ent_h = -(h*torch.log(1e-6 + h) + (1-h)*torch.log(1e-6 + 1 - h)).sum()
        
        # Compute ELBO
        ELBO = log_pmu + log_a_z + log_palpha + log_ph + log_z_az + log_pc + \
               log_pz1 + ent_z + ent_alpha + ent_mu + ent_c + ent_h
        
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, h, alpha):
        """
            gen_next_layer(ModelPoiFullScaled, ndarray, ndarray, ndarray, ndarray) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            h: (input_dim,) Change probability for layer t
            alpha: (embedding_dim,) New community center for layer t
            
            Returns:
                out: The predicted output probabilities for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)) ** 2).sum(dim=2)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-self.compute_scaled_dist(dist))
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        #h = h.unsqueeze(1).expand(n, d)
        #alpha = alpha.unsqueeze(0).expand(n, d)
        #Z_next = (1-h) * Z_next + h * alpha
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)) ** 2).sum(dim=2)
        a_preds = self.compute_output(dist2)
        
        return a_preds
        
        
                
        
        
        
        
        
class ModelFullScaled2(nn.Module):
    def __init__(self, num_nodes, K, hidden_dim=32, embedding_dim=8, \
                 int_mu_dim=8, int_alpha_dim=8, num_layers=1, \
                 use_bias_s1=False, use_bias_s2=False, use_bias_s3=False):
        """
            __init__(ModelFullScaled2, int, int, int, int, int, int, bool, \
                                                    bool, bool) -> None
            Implements the full LEM model with a Bi-Directional LSTM.
            Output matrix is binary.
            No neural network for output prediction and h prediction.
            
            num_nodes: Number of nodes in the network
            hidden_dim: Dimension of hidden state in LSTM
            embedding_dim: Dimension of the embedding space
            int_mu_dim: Dimension of intermediate layer for mu (layer 1)
            int_alpha_dim: Dimension of intermediate layer for alpha
            num_layers: Number of layers in the LSTM
            use_bias_s1: Whether to use bias in output scale calculation
            use_bias_s2: Whether to use bias in influence scale calculation
            use_bias_s3: Whether to use bias in influence scale calculation for
                         a new community
        """
        super(ModelFullScaled2, self).__init__()
        
        # Initialize dimensions
        self.input_dim = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.int_mu_dim = int_mu_dim
        self.int_alpha_dim = int_alpha_dim
        
        # Set up the lstm
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=\
                                            self.num_layers, bidirectional=True)
        self.lstm_hidden = self.init_lstm_hidden()
        
        # Set up linear layers for original mean and variance
        self.mean_orig = nn.Linear(self.hidden_dim*2, self.embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim*2, self.embedding_dim)

        # Set up intermediate layers for alphas and layer one mus
        self.alpha_int = nn.Linear(self.hidden_dim*2, self.int_alpha_dim)
        self.mu_int = nn.Linear(self.hidden_dim*2, self.int_mu_dim)

        # Set up alphas and layer one mus calculation layers
        self.alpha_m = nn.Linear(self.input_dim * self.int_alpha_dim, \
                               self.embedding_dim)
        self.alpha_log_s = nn.Linear(self.input_dim * self.int_alpha_dim, \
                               self.embedding_dim)
        self.mu_m = nn.Linear(self.input_dim * self.int_mu_dim, \
                               K * self.embedding_dim)
        self.mu_log_s = nn.Linear(self.input_dim * self.int_mu_dim, \
                               K * self.embedding_dim)
        
        # Set up h calculation layer
        self.h = nn.Linear(self.hidden_dim*2, 1)
        
        # Set up output prediction layer
        self.a_pred = nn.Linear(1, 1, bias=use_bias_s1)
        
        # Set up scaled distance layer for influence
        self.sc_dist_inf = nn.Linear(1, 1, bias=use_bias_s2)
        
        # Set up scaled distance layer for influence for new community
        self.sc_dist_new = nn.Linear(1, 1, bias=use_bias_s3)
        
        
        
    def init_lstm_hidden(self):
        """
            init_lstm_hidden(ModelFullScaled2) -> tuple
            Initializes the hidden state of LSTM
            
            Returns:
                (Variable, Variable): Zero hidden state for LSTM
        """
        return (Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)), \
                Variable(torch.zeros(self.num_layers*2, self.input_dim, \
                                     self.hidden_dim)))
    
    
    
    def forward(self, A, K, pi, s4=0.5):
        """
            forward(ModelFullScaled2, ndarray, ndarray, int, float) -> \
                (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, \
                 ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, \
                 ndarray, ndarray)
            
            A: (input_dim, input_dim, time_steps) Adjacency tensor of Dynamic
               network
            K: Number of communities in the first layer
            pi: (K,) Prior distribution over class
            s4: Scale parameter for sampling embedding from mean
            
            Returns:
                mean_orig: (input_dim, embedding_dim, num_steps) Original \
                            means
                log_vars: (input_dim, embedding_dim, num_steps) Log \
                           variance for embeddings
                m_alpha: (embedding_dim, num_steps) Alpha mean for each \
                          step
                s_alpha: (embedding_dim, num_steps) Alpha log var for \
                          each time step
                alphas: (embedding_dim, num_steps) Sampled alphas
                h: (input_dim, num_steps) Change probability for each node
                m_mus: (K, embedding_dim) Mu mean
                s_mus: (K, embedding_dim) Mu log var
                mus: (K, embedding_dim) Sampled mus
                c: (input_dim, K) Class assignment probability for first
                    layer
                a_preds: (input_dim, input_dim, num_steps) Output preds
                sc_dist_inf: (input_dim, input_dim, num_steps) Scaled distance
                             for influence prediction
                sc_dist_new: (input_dim, num_steps) Scaled distance
                             for new community influence prediction
                mean_final: (input_dim, embedding_dim, num_steps) Mean for final
                            embeddings
                embeddings: (input_dim, embedding_dim, num_steps) Sampled \
                            embeddings
        """
        # Bring A in correct form
        A = A.permute(2, 0, 1).contiguous()
        num_steps = A.size()[0]
        
        # Run the lstm
        out, self.lstm_hidden = self.lstm(A, self.lstm_hidden)
        
        # Initialize the required variables
        mean_orig = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]
        log_vars = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]

        m_mus = Variable(torch.Tensor(K, self.embedding_dim))
        s_mus = Variable(torch.Tensor(K, self.embedding_dim))
        mus = Variable(torch.Tensor(K, self.embedding_dim))

        m_alpha = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]
        s_alpha = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]
        alphas = [Variable(torch.Tensor(self.embedding_dim)) \
                       for _ in range(num_steps)]

        c = Variable(torch.Tensor(self.input_dim, K))

        h = [Variable(torch.Tensor(self.input_dim)) \
                       for _ in range(num_steps)]

        a_preds = [Variable(torch.Tensor(self.input_dim, self.input_dim)) \
                       for _ in range(num_steps)]
                       
        sc_dist_inf = [Variable(torch.Tensor(self.input_dim, self.input_dim)) \
                       for _ in range(num_steps)]
        
        sc_dist_new = [Variable(torch.Tensor(self.input_dim)) \
                       for _ in range(num_steps)]
        
        mean_final = [Variable(torch.Tensor(self.input_dim, \
                       self.embedding_dim)) for _ in range(num_steps)]
        
        embeddings = [Variable(torch.Tensor(self.input_dim, \
                      self.embedding_dim)) for _ in range(num_steps)]

        # Get values for the variables
        for i in range(num_steps):
            # Get the original mean
            mean_orig[i] = self.mean_orig(out[i, :, :])

            # Get the variance for q(z)
            log_vars[i] = self.log_var(out[i, :, :])
            
            # Add the effect of splitting
            if i > 0:   # Take splitting into account from second layer onwards
                # Alpha and h would have been calculated at previous timestep
                alpha1 = alphas[i-1].unsqueeze(0).expand(self.input_dim, \
                                                    self.embedding_dim)
                curr_h = h[i-1].unsqueeze(1).expand(self.input_dim, \
                                                    self.embedding_dim)
                
                
                # Find emebddings taking into account mean_orig, alpha and h
                temp2 = alpha1 * curr_h + (1 - curr_h) * mean_orig[i]
                mean_final[i] = temp2
                embeddings[i] = temp2 + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                           torch.exp(log_vars[i]/2)
                  
            else:   # No splitting in first layer, so without split mean is used
                mean_final[i] = mean_orig[i]
                embeddings[i] = mean_orig[i] + \
                  Variable(torch.randn(self.input_dim, self.embedding_dim)) * \
                           torch.exp(log_vars[i]/2)
            
            # Calculate alphas and h, should be done for all layers
            temp = F.tanh(self.alpha_int(out[i, :, :])).view(1, -1)
            m_alpha[i] = self.alpha_m(temp).view(-1)
            s_alpha[i] = self.alpha_log_s(temp).view(-1)
            alphas[i] = m_alpha[i] + \
                        Variable(torch.randn(self.embedding_dim)) * \
                                 torch.exp(s_alpha[i]/2)

            alpha = alphas[i].expand(self.input_dim, self.embedding_dim)
            alpha = ((embeddings[i] - alpha) ** 2).sum(dim=1).view(-1, 1)
            sc_dist_new[i] = torch.abs(self.sc_dist_new(alpha).view(-1))
            
            h[i] = F.sigmoid(self.h(out[i, :, :])).view(-1)
            
            # Compute output prediction for each layer
            dist = embeddings[i].unsqueeze(0).expand(self.input_dim, \
                                        self.input_dim, self.embedding_dim)
            dist = ((dist - dist.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
            a_preds[i] = 1 - torch.tanh(torch.abs(self.a_pred(dist).view(\
                                      self.input_dim, self.input_dim)))
            sc_dist_inf[i] = torch.abs(self.sc_dist_inf(dist)).view(\
                                      self.input_dim, self.input_dim)
            
            # Compute mus and c for first layer      
            if i == 0:
                temp3 = F.tanh(self.mu_int(out[i, :, :])).view(1, -1)
                m_mus = self.mu_m(temp3).view(K, self.embedding_dim)
                s_mus = self.mu_log_s(temp3).view(K, self.embedding_dim)
                mus = m_mus + \
                  Variable(torch.randn(K, self.embedding_dim)) * \
                           torch.exp(s_mus/2)

                dist1 = embeddings[i].unsqueeze(0).expand(K, self.input_dim, \
                                                        self.embedding_dim)
                dist1 = torch.sum((dist1 - mus.unsqueeze(0).expand(\
                            self.input_dim, K, self.embedding_dim).\
                                     permute(1, 0, 2))**2, dim=2).t()
                dist1 = torch.exp(-dist1 / (2 * s4**2))
                ditd1 = dist1 * pi.unsqueeze(0).expand(self.input_dim, K)
                c = (1e-6 + dist1) / (K * 1e-6 + dist1.sum(dim=1)).\
                                    unsqueeze(1).expand(self.input_dim, K)
                

        # Convert lists to tensors   
        mean_orig = torch.cat([x.unsqueeze(2) for x in mean_orig], dim=2)
        log_vars = torch.cat([x.unsqueeze(2) for x in log_vars], dim=2)
        m_alpha = torch.cat([x.unsqueeze(1) for x in m_alpha], dim=1)
        s_alpha = torch.cat([x.unsqueeze(1) for x in s_alpha], dim=1)
        alphas = torch.cat([x.unsqueeze(1) for x in alphas], dim=1)
        h = torch.cat([x.unsqueeze(1) for x in h], dim=1)
        a_preds = torch.cat([x.unsqueeze(2) for x in a_preds], dim=2)
        sc_dist_inf = torch.cat([x.unsqueeze(2) for x in sc_dist_inf], dim=2)
        sc_dist_new = torch.cat([x.unsqueeze(1) for x in sc_dist_new], dim=1)
        mean_final = torch.cat([x.unsqueeze(2) for x in mean_final], dim=2)
        embeddings = torch.cat([x.unsqueeze(2) for x in embeddings], dim=2)
        
        return (mean_orig, log_vars, m_alpha, s_alpha, alphas, h, m_mus, \
                s_mus, mus, c, a_preds, sc_dist_inf, sc_dist_new, mean_final, \
                embeddings)
        

    
    
    def loss_function(self, A, embeddings, variances, a_preds, sc_dist_inf, \
                      sc_dist_new, alphas, mus, s_alpha, s_mus, c, h, pi, \
                      m=0.0, s=1.0, s4=0.5):
        """
            loss_function(Model, ndarray, ndarray, ndarray, ndarray, \
                          ndarray, ndarray, ndarray, ndarray, ndarray, \
                          ndarray, ndarray, ndarray, ndarray, float, \
                          float, float) -> float
            Computes the ELBO
            
            A: (input_dim, input_dim, num_steps) Adjacency tensor of Dynamic
               Network
            embeddings: (input_dim, embedding_dim, num_steps) Embeddings sampled
                        for each node
            variances: (input_dim, embedding_dim, num_steps) variances of em-
                        -beddings
            a_preds: (input_dim, input_dim, num_steps) Output probability
            sc_dist_inf: (input_dim, input_dim, num_steps) Scaled distance
                         for influence prediction
            sc_dist_new: (input_dim, num_steps) Scaled distance
                         for new community influence prediction
            alphas: (embedding_dim, num_steps) Sampled alphas
                    m_mus: (K, embedding_dim) Mu mean
            mus: (K, embedding_dim) Sampled mus
            s_alpha: (embedding_dim, num_steps) Alpha log var for \
                              each time step
            s_mus: (K, embedding_dim) Mu log var
            c: (input_dim, K) Class assignment probability for first
                        layer
            h: (input_dim, num_steps) Change probability for each node
            pi: (K,) Class membership prior for first layer
            m: (embedding_dim,) The mean for prior on z^(1). If 0 then zero 
                   vector is used
            s: Scale parameter for prior on w^(1)
            s4: Scale parameter for P(Z_t | Z_t-1, A_t-1)
                        
            Returns:
                ELBO: The value of negative elbo
        """
        # Some useful variables
        n, _, T = A.size()
        _, d, _ = embeddings.size()
        
        if m == 0.0:
            m = Variable(torch.zeros(n, d))
        else:
            m = Variable(m.unsqueeze(0).expand(n, d))
        K = pi.size()[0]

        # Compute log p(mu)
        log_pmu = -torch.sum((mus - Variable(torch.zeros(K, d))) ** 2) / (2*s**2)
                
        # Compute log p(alphas)
        log_palpha = -torch.sum((alphas - Variable(torch.zeros(d, T)))\
                      ** 2) / (2 * s**2)
        
        # Compute log P(c)
        log_pc = torch.sum(c * torch.log(1e-6 + pi.unsqueeze(0).expand(n, K)))
        
        # Compute distances between layer 1 embeddings and mus
        dist_l1 = embeddings[:, :, 0].unsqueeze(0).expand(K, n, d)
        dist_l1 = torch.sum((dist_l1 - mus.unsqueeze(0).expand(n, K, d).\
                             permute(1, 0, 2))**2, dim=2).t()
        dist_l1 = -dist_l1 / (2 * s4**2)
        log_pz1 = torch.sum(c * dist_l1)
        
        # Compute log p(h) for all layers
        log_ph = 1 - torch.tanh(sc_dist_new)
        log_ph = h * torch.log(1e-6 + log_ph) + \
                 (1-h) * torch.log(1e-6 + 1-log_ph)
        log_ph = log_ph.sum()
        
        # Compute prod_t P(z^(t) | z^(t-1), A^(t-1), h(t), alpha(t))
        d2 = torch.exp(-sc_dist_inf)
        log_z_az = 0.0
        for t in range(1, T):
            means = torch.matmul(d2[:, :, t-1] * (A[:, :, t-1] + \
                    Variable(torch.eye(n))), embeddings[:, :, t-1]) / \
                    (1e-6 + (d2[:, :, t-1] * (A[:, :, t-1]+Variable(torch.eye(n))))\
                    .sum(dim=1)).unsqueeze(1).expand(n, d)
            
            log_z_az -= (((1-h[:, t-1]) * ((embeddings[:, :, t] - \
                                        means) ** 2).sum(dim=1)) / (2 * s4**2) + \
                        (h[:, t-1] * ((embeddings[:, :, t] - \
                                        alphas[:, t-1].unsqueeze(0).expand(n, d)) \
                                        ** 2).sum(dim=1)) / (2 * s4**2)).sum()
        
        # Compute  prod_t P(A^(t) | z^(t))
        log_a_z = torch.triu((A * torch.log(1e-6 + a_preds) + (1-A) * \
                              torch.log(1e-6 + 1 - a_preds)).\
                             sum(dim=2), diagonal=1).sum()
        
        # Compute the entropy term for z's
        ent_z = torch.sum(variances) / 2
        
        # Compute the entropy term for alpha's
        ent_alpha = torch.sum(s_alpha) / 2
        
        # Compute the entropy term for mus's
        ent_mu = torch.sum(s_mus) / 2        
        
        # Compute the entropy term for c
        ent_c = -(c * torch.log(1e-6 + c)).sum()
        
        # Compute entropy for hs
        ent_h = -(h*torch.log(1e-6 + h) + (1-h)*torch.log(1e-6 + 1 - h)).sum()
        
        # Compute ELBO
        ELBO = log_pmu + log_a_z + log_palpha + log_ph + log_z_az + log_pc + \
               log_pz1 + ent_z + ent_alpha + ent_mu + ent_c + ent_h
        
        return -ELBO
        
        
    def gen_next_layer(self, A, embeddings, h, alpha):
        """
            gen_next_layer(Model, ndarray, ndarray, ndarray, ndarray) -> ndarray
            
            Predicts the adjacency matrix entries in the layer t
            
            A: (input_dim, input_dim) Observed adjacency matrix for layer t - 1
            embeddings: (input_dim, embedding_dim) Embeddings for layer t - 1
            h: (input_dim,) Change probability for layer t
            alpha: (embedding_dim,) New community center for layer t
            
            Returns:
                out: The predicted output probabilities for layer t
        """
        # Some useful variables
        n = A.size()[0]
        d = embeddings.size()[1]
        
        # Compute the distances between different points in latent space
        dist = embeddings.unsqueeze(0).expand(self.input_dim, \
                                            self.input_dim, self.embedding_dim)
        dist = ((dist - dist.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
        
        # Compute the next layer embeddings mean
        d2 = torch.exp(-torch.abs(self.sc_dist_inf(dist)).view(\
                                      self.input_dim, self.input_dim))
        Z_next = torch.matmul(d2 * (A + Variable(torch.eye(n))), embeddings) / \
                 (1e-6 + (d2 * (A + Variable(torch.eye(n)))).sum(dim=1))\
                .unsqueeze(1).expand(n, d)
        #h = h.unsqueeze(1).expand(n, d)
        #alpha = alpha.unsqueeze(0).expand(n, d)
        #Z_next = (1-h) * Z_next + h * alpha
        
        # Compute the distances between updated embeddings                   
        dist2 = Z_next.unsqueeze(0).expand(n, n, d)
        dist2 = ((dist2 - dist2.permute(1, 0, 2)).view(-1, \
                              self.embedding_dim) ** 2).sum(dim=1).view(-1, 1)
        a_preds = 1 - F.tanh(torch.abs(self.a_pred(dist2).view(\
                                      self.input_dim, self.input_dim)))
        
        return a_preds

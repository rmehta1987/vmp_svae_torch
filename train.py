import time
from gmmsvae import GMMSVAE
import torch
import argparse
from visdom import Visdom
from torchvision import datasets, transforms
from torch import nn
from torch_utils import exponential_learning_rate
from torch.utils.tensorboard import SummaryWriter
from torch_utils import make_pinwheel_data
from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser(description='SVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--nb_components', type=int, default=15, metavar='N',
                    help='how many mixture componenents')
parser.add_argument('--latent_dims', type=int, default=2, metavar='N',
                    help='how many latent dimensions')
                    
args = parser.parse_args()
args.device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
args.nb_samples = 10 # Number of samples per latent component for the Mixture Model

device = torch.device("cuda" if args.device else "cpu")


kwargs = {'num_workers': 0, 'pin_memory': True} if args.device else {}


# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

# Create Pin Wheel dataset
num_clusters = 5                            # number of clusters in pinwheel data
samples_per_cluster = 200                   # number of samples per cluster in pinwheel
K = args.nb_components                      # number of components in mixture model
N = args.latent_dims                        # number of latent dimensions
P = 2                                       # number of observation dimensions
data, labels = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

# Convert into pytorch dataloader paradigm
dataset = TensorDataset(data, labels)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, **kwargs)

# decoder_type = 'bernoulli' if config['dataset'] in ['mnist', 'mnist-small'] else 'standard'
decoder_type = 'standard'
# encoder_layers = [(100, nn.Tanh()), (100, nn.Tanh()), (args.latent_dims, 'natparam')]
# decoder_layers = [(100, nn.Tanh()), (100, nn.Tanh()), (2, decoder_type)]
encoder_layers = [(40, nn.Tanh()), (40, nn.Tanh()), (args.latent_dims, 'natparam')]
decoder_layers = [(40, nn.Tanh()), (40, nn.Tanh()), (P, decoder_type)]

# Tensorboard
writer = SummaryWriter()

model = GMMSVAE(args, encoder_layers, decoder_layers, 2).to(device)
the_theta = model.theta
the_gmm_prior = model.gmm_prior
optim = torch.optim.Adam(model.parameters(), lr=.0003)
lrcvi = 0.2 # cvi stepsize
decay_rate = 0.95
decay_steps = 1000
global_step = 1
for epoch in range(args.epochs):
    epoch_start_time = time.time()
    iter_count = 0

    try:
        print ("Experiment at epoch {}".format(epoch))

        for i, (data, _) in enumerate(train_loader):
            batch_start_time = time.time()
            y_reconstruction, x_given_y_phi, x_k_samples, x_samples, log_z_given_y_phi, phi_gmm, phi_tilde = model(data)
            #elbo, _ = model.compute_elbo(data, y_reconstruction, the_theta, phi_tilde, x_k_samples, log_z_given_y_phi, decoder_type)
            elbo = model.compute_elbo_debug(data, y_reconstruction, the_theta, phi_tilde, x_k_samples, log_z_given_y_phi, decoder_type)
            if epoch == 0:
                writer.add_graph(model,data)
            # Update GMM parameters
            if i == 0:
                theta_star = model.m_step(gmm_prior=the_gmm_prior, x_samples=x_samples,
                                                        r_nk=torch.exp(log_z_given_y_phi))
                the_theta = model.update_gmm_params(the_theta, theta_star, lrcvi)
                lrcvi = exponential_learning_rate(lrcvi, decay_rate, global_step, decay_steps)
            else:
                with torch.no_grad():
                    theta_star = model.m_step(gmm_prior=the_theta, x_samples=x_samples,
                                                            r_nk=torch.exp(log_z_given_y_phi))
                    the_theta = model.update_gmm_params(the_theta, theta_star, lrcvi)
                    lrcvi = exponential_learning_rate(lrcvi, decay_rate, global_step, decay_steps)

            if i % 10 == 0:
                # keep some parameters for debugging
                phi_gmm2 = phi_gmm
                yrecon = y_reconstruction
            global_step += 1
            # Optimize NN parameters
            elbo.backward()
            optim.step()
            model.zero_grad()
            
           
            #Check elbo:
            if i % 10 == 0:
                neg_normed_elbo = elbo.item() / len(data)
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, i*len(data), len(train_loader.dataset), neg_normed_elbo))
                

        epoch_end_time = time.time()

        print("Finished Epoch in {:.4f} seconds at epoch: {}".format(epoch_start_time-epoch_end_time,epoch))
        
    except Exception as e:
        writer.close()
        print("Crashed")
        print(e)
        print("Crashed at Epoch: {}".format(epoch))




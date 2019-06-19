import time
from gmmsvae import GMMSVAE
import torch
import argparse
from visdom import Visdom
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description='SVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=600, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

#decoder_type = 'bernoulli' if config['dataset'] in ['mnist', 'mnist-small'] else 'standard'
decoder_type = 'standard'
encoder_layers = [(100, nn.Tanh()), (100, nn.Tanh()), (10, 'natparam')]
decoder_layers = [(100, nn.Tanh()), (100, nn.Tanh()), (784, decoder_type)]

model = GMMSVAE(args, encoderlayers, decoderlayers).to(device)
the_theta = model.theta
the_gmm_prior = model.gmm_prior
optim = torch.optim.Adam(model.parameters(), lr=.0003)
optim
for epoch in range(args.epochs):
    epoch_start_time = time.tim()
    iter_count = 0

    for i, data in enumerate(train_loader):
        batch_start_time = time.time()
        y_reconstruction, x_given_y_phi, x_k_samples, x_samples, 
        log_z_given_y_phi, phi_gmm, phi_tilde = model(data)
        elbo, details = model.compute_elbo(y_reconstruction, the_theta, phi_tilde,
                                                                  x_k_samples, log_z_given_y_phi,
                                                                  decoder_type=self.decoder_type)
        # Update GMM parameters
        if i == 1:
            theta_star = GMMSVAE.m_step(gmm_prior=the_gmm_prior, x_samples=x_samples,
                                                    r_nk=torch.exp(log_z_given_y_phi))
            the_theta = GMMSVAE.update_gmm_params(the_theta, theta_star, lrcvi)
        else:
            theta_star = GMMSVAE.m_step(gmm_prior=the_theta, x_samples=x_samples,
                                                    r_nk=torch.exp(log_z_given_y_phi))
            the_theta = GMMSVAE.update_gmm_params(the_theta, theta_star, lrcvi)

        # Optimize NN parameters
        elbo *= -1
        elbo.backward()
        optim.step()
        optim.zero_grad()
        
        #Check elbo:
        if i % 100 == 0:
            neg_normed_elbo = torch.divide(torch.sum(elbo.item()), len(data))
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, i*len(data), len(train_loader.dataset), neg_normed_elbo))

        batch_end_time = time.time()

        print ("Finished 1st batch in {:.4f} seconds".format(batch_end_time-batch_start_time))




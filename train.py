import time
from gmmsvae import GMMSVAE
import torch
import argparse
from visdom import Visdom
from torchvision import datasets, transforms
from torch import nn
from torch_utils import exponential_learning_rate
from torch.utils.tensorboard import SummaryWriter
#rom tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset
from torch_utils import make_pinwheel_data
from torchviz import make_dot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import niw
import gmm
import dirichlet
import gaussian
import numpy as np
from losses import diagonal_gaussian_logprob, weighted_mse
import traceback
from plots import plot_clustered_data, plot_clusters 
from torchvision.utils import save_image
import sys # for debugging crashes


## Add these to visualize gradients
# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/17
# https://discuss.pytorch.org/t/how-to-check-for-vanishing-exploding-gradients/9019


def plot_grad_flow(named_parameters, global_step):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    ave_grads_prior = []
    max_grads_prior= []
    layers_prior = []
    for n, p in named_parameters:
        if(p.requires_grad) and (".b" not in n) and ("train" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
        elif(p.requires_grad) and (".b" not in n) and ("train" in n):
            layers_prior.append(n)
            ave_grads_prior.append(p.grad.abs().mean())
            max_grads_prior.append(p.grad.abs().max())

    #plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")

    #plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.plot(max_grads, alpha=0.3, lw=1, color="r")
    plt.plot(ave_grads, alpha=0.3, lw=1, color="b")
    plt.tight_layout() 
    plt.savefig("plot_grad_flow_{}.svg".format(global_step))
    plt.close()
    '''
    #plot prior parameters gradients
    plt.bar(np.arange(len(max_grads_prior)), max_grads_prior, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_grads_prior)), ave_grads_prior, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads_prior)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads_prior), 1), layers_prior, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads_prior))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("prior_plot_grad_flow_{}.svg".format(global_step))
    plt.close()
    '''

parser = argparse.ArgumentParser(description='SVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--nb_components', type=int, default=15, metavar='N',
                    help='how many mixture componenents')
parser.add_argument('--latent_dims', type=int, default=4, metavar='N',
                    help='how many latent dimensions')
parser.add_argument('--nb_samples', type=int, default=50, metavar='N',
                    help='how many samples per mixture component')                    
args = parser.parse_args()
args.device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

device = torch.device("cuda" if args.device else "cpu")


kwargs = {'num_workers': 0, 'pin_memory': True} if args.device else {}  # Need to set num_workers 0 in vs_code, as it cannot create multiple processes

'''
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
'''

# Create Pin Wheel dataset ------------------------------------------------------
num_clusters = 5                            # number of clusters in pinwheel data
samples_per_cluster = 300                   # number of samples per cluster in pinwheel
K = args.nb_components                      # number of components in mixture model
N = args.latent_dims                        # number of latent dimensions
P = 2                                       # number of observation dimensions
data, labels = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

# Convert into pytorch dataloader paradigm
args.batch_size=100
dataset = TensorDataset(data, labels)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1400, 100])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

# Create a test-subset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, **kwargs)
# End Pin Wheel dataset ------------------------------------------------------



# decoder_type = 'bernoulli' if config['dataset'] in ['mnist', 'mnist-small'] else 'standard'
decoder_type = 'standard'

#encoder_layers = [(100, nn.Tanh()), (100, nn.Tanh()), (args.latent_dims, 'natparam')]
#decoder_layers = [(100, nn.Tanh()), (100, nn.Tanh()), (784, decoder_type)]  # How to figure out input dimensions of data ???


### Auto DataSet

# encoder_layers = [(100, nn.Tanh()), (100, nn.Tanh()), (args.latent_dims, 'natparam')]
# decoder_layers = [(100, nn.Tanh()), (100, nn.Tanh()), (2, decoder_type)]

## PinWheel Dataset
encoder_layers = [(50, nn.Tanh()), (50, nn.Tanh()), (args.latent_dims, 'natparam')]
decoder_layers = [(50, nn.Tanh()), (50, nn.Tanh()), (P, decoder_type)]

## PinWheel Dataset, but only using regular VAE
#encoder_layers = [(50, nn.Tanh()), (50, nn.Tanh()), (args.latent_dims, nn.Tanh())]
#decoder_layers = [(50, nn.Tanh()), (50, nn.Tanh()), (P, decoder_type)]


model = GMMSVAE(args, encoder_layers, decoder_layers, P).to(device)  # How to figure out input dimensions of data ???
the_theta = model.theta
the_gmm_prior = model.gmm_prior
optim = torch.optim.Adam(model.parameters(), lr=.01)
lrcvi = 0.1 # cvi stepsize
decay_rate = 1
decay_steps = 1000
global_step = 1
phi_gmm2 = []
writer = SummaryWriter(comment='Histgorams')
for epoch in range(args.epochs):
    epoch_start_time = time.time()
    iter_count = 0

    try:
        print ("Experiment at epoch {}".format(epoch))

        for i, (data, _) in enumerate(train_loader):

            batch_start_time = time.time()
            data = data.to(device)
            y_reconstruction, x_given_y_phi, x_k_samples, x_samples, log_z_given_y_phi, phi_gmm, phi_tilde = model(data) # model(data.view(-1, 784))
            elbo, details = model.compute_elbo(data, y_reconstruction, the_theta, phi_tilde, x_k_samples, log_z_given_y_phi, decoder_type) #model.compute_elbo(data.view(-1, 784), y_reconstruction, the_theta, phi_tilde, x_k_samples, log_z_given_y_phi, decoder_type)
            # elbo = model.compute_elbo_debug(data, y_reconstruction, the_theta, phi_tilde, x_k_samples, log_z_given_y_phi, decoder_type)
            # Update GMM parameters
            #if epoch == 0 and i == 0:  # Creating Graphical Visualization of Pytorch Computational Graph
            #    dot = make_dot(elbo, params=dict(model.named_parameters()))
            #    dot.render('test-output/round-table_2.gv', view=True)
            #    sys.exit(1)
            #if epoch == 0 and i == 0:
            #    with SummaryWriter(comment='GraphModel_1') as w:
            #        w.add_graph(model, data, False)
            #        w.close()
            #        sys.exit(1)
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

            
            global_step += 1
            # Optimize NN parameters
            elbo.backward()
            optim.step()

            if i % 10 == 0:
                # keep some parameters for debugging
                phi_gmm_unpacked = model.unpack_recognition_gmm(phi_gmm)
                u_mu, u_cov = gaussian.natural_to_standard(phi_gmm_unpacked[0],phi_gmm_unpacked[1])
                #wmse = weighted_mse(data, y_reconstruction[0].detach().cpu(), torch.exp(log_z_given_y_phi).detach().cpu())
                #print ("Training wmse {}".format(wmse))
                #glogliki = diagonal_gaussian_logprob(data.view(-1, 784).to(device), y_reconstruction[0].detach(), y_reconstruction[1].detach(), log_z_given_y_phi.detach())
                #print ("Training diagonal gaussian logprob {}".format(glogliki))
                #if i % 100 == 0:
                    #plot_grad_flow(model.named_parameters(), global_step)
                if i % 100 == 0:
                    writer.add_histogram('logz', torch.exp(log_z_given_y_phi), global_step=global_step)
                    for index, (name, kernel) in enumerate(model.named_parameters()):
                        writer.add_histogram('{}_grad'.format(name), kernel.grad, global_step=global_step)
                    writer.add_embedding(u_mu,tag='mu_phi_gmm', global_step=global_step)
                    #writer.add_embedding(u_cov,tag='cov_phi_gmm')
                    writer.add_histogram('pi_phi_gmm',torch.exp(phi_gmm_unpacked[-1]), global_step=global_step)
                  
                    beta_k, m_k, C_k, v_k = niw.natural_to_standard(the_theta[1], the_theta[2], the_theta[3], the_theta[4])
                    mu, sigma = niw.expected_values((beta_k, m_k, C_k, v_k))
                    alpha_k = dirichlet.natural_to_standard(the_theta[0])
                    expected_log_pi = dirichlet.expected_log_pi(alpha_k)
                    pi_theta = torch.exp(expected_log_pi)
                    writer.add_embedding(mu,tag='mu', global_step=global_step)
                    #writer.add_embedding(cov,tag='cov')
                    writer.add_histogram('pi_theta', pi_theta,  global_step=global_step)




            model.zero_grad()


                
            
           
            #Check elbo:
            #if i % 10 == 0:
            #    neg_normed_elbo = elbo.item() / len(data)
            #    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, i*len(data), len(train_loader.dataset), neg_normed_elbo))
                

        epoch_end_time = time.time()

        print("Finished Epoch in {:.4f} seconds at epoch: {}".format(epoch_end_time-epoch_start_time,epoch))
        neg_normed_elbo = elbo.item() / len(data)
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, i*len(data), len(train_loader.dataset), neg_normed_elbo))
        print("Regualizer Term: {}".format(details[-1].item() / len(data)))
        
    except Exception as e:
        print("Crashed")
        print(e)
        print(traceback.format_exc())
        print("Crashed at Epoch: {}".format(epoch))
        sys.exit(1)


print ("creating Plots")

# Test trained model
model.eval()
testset, the_org_cluster = iter(test_loader).next()
y_reconstruction, x_given_y_phi, x_k_samples, x_samples, log_z_given_y_phi, phi_gmm, phi_tilde = model(testset.to(device)) #model(testset.view(-1, 784))
y_mean = y_reconstruction[0].clone().cpu()

# #### Plot Clusters

# # Find most optimal cluster allocation
# 
cluster_comp = torch.max(log_z_given_y_phi,dim=1)[1].clone().cpu()  # torch returns [values, indicies]
N, D = data.shape
n_idx = torch.arange(start=0,end=N).reshape(-1,1).to(torch.int64) # Shape [N x 1]
s_idx = torch.zeros(N).reshape(-1,1).to(torch.int64) # Shape [N x 1]
d_idx = torch.arange(start=0,end=D)[None,:,None] # Shape: [1 x D x 1]
k_idx = cluster_comp.reshape(-1,1)

# wmse = weighted_mse(testset.view(-1, 784), y_mean, torch.exp(log_z_given_y_phi).detach().cpu())
# print ("Test wmse {}".format(wmse))
# '''
# # Plot Weighted Reconstruction
# weighted_recon = torch.mean(y_mean.detach().cpu(),dim=2)

# # weight mse with predicted component responsibilities
# # sum over (weighted) compontents; shape = N
# # mean over data points; shape = 1
# y_weight_recon = torch.sum(weighted_recon*torch.exp(log_z_given_y_phi).detach().cpu().unsqueeze(-1),dim=1)
# # plt.scatter(testset[:, 0], testset[:, 1], color='lightgray', s=size_datapoint_tr)
# # plt.scatter(y_weight_recon[:, 0], y_weight_recon[:, 1], color='blue', s=size_datapoint_tr)
# #plt.title('Weighted Recon')
# #plt.savefig('Weighted_minst.svg')


# ### For image datasets
# #n = 8
# #comparison = torch.cat([testset[:n], y_weight_recon.view(args.batch_size, 1, 28, 28)[:n]])
# #save_image(comparison.cpu(), 'comp_recon.png', nrow=n)
# #save_image(testset[0].view(1, 28, 28), 'org_epoch_15_lr_003_ld_8.png')
# #save_image(y_weight_recon[0].view(1,28,28), 'recong_epoch_15_lr_003_ld_8.png')
# ###


# ####  Add create a label to the reconstructed Y_MEANS
# #plot_clustered_data(testset, y_weight_recon, the_org_cluster, ax=None)
# #plt.savefig('Test_weighted_minst.svg')




y_mean_rec = y_mean[[n_idx, k_idx, s_idx]].squeeze(1) # Shape [N x D]
f, ax = plt.subplots(2,2)


beta_k, m_k, C_k, v_k = niw.natural_to_standard(the_theta[1], the_theta[2], the_theta[3], the_theta[4])
mu, sigma = niw.expected_values((beta_k, m_k, C_k, v_k))
alpha_k = dirichlet.natural_to_standard(the_theta[0])
expected_pi = dirichlet.expected_log_pi(alpha_k)
pi_theta = torch.exp(expected_pi).detach().cpu().numpy()
q_z = torch.exp(log_z_given_y_phi.clone().detach())

plot_clustered_data(data.cpu(), y_mean_rec.detach(), cluster_comp.detach(), ax=ax[0,0])

plot_clusters(x_samples.detach().cpu().numpy(), mu.detach().cpu().numpy(), sigma.detach().cpu().numpy(), q_z.detach().cpu(), pi_theta, ax=ax[1,0], title='Latent Space')

# ordered colours (neighbouring colours differ strongly)
colours = ('b', 'g', 'r', 'c', 'm', 'navy', 'lime', 'y', 'k', 'pink', 'orange', 'magenta', 'firebrick', 'olive',
          'aqua', 'sienna', 'khaki', 'teal', 'darkviolet', 'darkseagreen')

ax[0, 1].bar(np.arange(len(pi_theta)), pi_theta, color=colours)


#torch.save(phi_gmm2, 'phi_gmm_unpacked_3.pt')

plt.savefig('test_torch_100_latent_10_lr_01_latent_dim_4.svg')


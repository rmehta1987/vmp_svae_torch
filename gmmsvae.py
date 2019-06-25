import numpy as np
import torch
from Decoder import Decoder
from Encoder import Encoder
from torch_utils import init_tensor_gpu_grad
from torch import nn
import niw
import gmm
import dirichlet
import gaussian


class GMMSVAE(nn.Module):
    def __init__(self, opts, encoderlayers, decoderlayers, input_dim=784):
        super(GMMSVAE, self).__init__()
        self.device = opts.device
        self.encoder_layers = encoderlayers
        self.decoder_layers = decoderlayers
        self.decoder_type = decoderlayers[-1][1]
        self.nb_components = opts.nb_components
        self.nb_samples = opts.nb_samples
        self.latent_dims = opts.latent_dims
        self.seed = opts.seed
        self.input_dim = input_dim
        self.x_given_y_phi_model = Encoder(self.encoder_layers,input_dim)
        self.y_reconstruction_model = Decoder(self.decoder_layers,self.latent_dims) # [type of layers, input_dim]
        self.gmm_prior, self.theta = self.init_mm(self.nb_components, self.latent_dims, self.seed, self.device)
        #self.pi_k = torch.nn.Parameter(torch.randn((self.nb_components,),requires_grad=True).to(self.device))
        self.pi_k = torch.randn((self.nb_components,),requires_grad=False).to(self.device) # see if this is causing backwards problem
        self.train_mu_k, self.train_L_k, self.soft_pi_k = self.init_recognition_params(self.theta, self.nb_components, self.seed, self.device)
        self.train_pi_k = self.soft_pi_k(self.pi_k)
        #self.phi_gmm = (self.train_mu_k, self.train_L_k, self.train_pi_k)
        self.totiter = 0

    
    def init_mm(self, nb_components, latent_dims, seed=0, param_device='cuda', theta_as_variable=True):
        '''
        Args:
        Returns:
            theta: Contains hyperparameters [alpha, A, b, beta, v_hat] where A, b, beta, v_hat are
            parameters of the NIW prior and alpha is the dirichlet prior on the parameters
        '''

        # prior parameters area always constance so the don't take gradients for them
        theta_prior = self.init_mm_params(nb_components, latent_dims, alpha_scale=0.05 / nb_components, beta_scale=0.5, 
                                    m_scale=0, C_scale=latent_dims + 0.5, v_init=latent_dims + 0.5,seed=0,
                                    as_variables=False, trainable=False, device=param_device)

        theta = self.init_mm_params(nb_components, latent_dims, alpha_scale=1., beta_scale=1., m_scale=5.,
                                C_scale=2 * (latent_dims), v_init=latent_dims + 1., seed=0,
                                as_variables=theta_as_variable, trainable=False)
        
        return theta_prior, theta

    def init_mm_params(self, nb_components, latent_dims, alpha_scale=.1, beta_scale=1e-5, v_init=10., m_scale=1., C_scale=10.,
                    seed=0, as_variables=True, trainable=False, device='cuda'):
        
        alpha_init = alpha_scale * torch.ones(nb_components,) # shape [nb_components]
        beta_init = beta_scale * torch.ones(nb_components,)  # shape [nb_components]
        v_init = torch.tensor([float(latent_dims + v_init)]).expand(nb_components)  # shape [nb_components]
        means_init = m_scale * torch.empty(nb_components, latent_dims).uniform_(-1,1)  # shape [nb_components, latent_dims]  - uniform random matrix between -1 to 1
        covariance_init = C_scale * torch.eye(latent_dims).expand(nb_components, -1, -1)  # shape nb_components x latent_dims x latent_dims 

        A, b, beta, v_hat = niw.standard_to_natural(beta_init, means_init, covariance_init, v_init)
        alpha = dirichlet.standard_to_natural(alpha_init)

        if as_variables:
            alpha = init_tensor_gpu_grad(alpha, trainable=trainable, device=device)
            A = init_tensor_gpu_grad(A, trainable=trainable, device=device)
            b = init_tensor_gpu_grad(b, trainable=trainable, device=device)
            beta = init_tensor_gpu_grad(beta, trainable=trainable, device=device)
            v_hat = init_tensor_gpu_grad(v_hat, trainable=trainable, device=device)
        
        return alpha, A, b, beta, v_hat


    def init_recognition_params(self, theta, nb_components, seed=0, param_device='cuda'):
        '''
        Args:
            theta [is a tuple that contains the following parameters]:
                alpha - the weights of the mixtures
                A - natural parameter of NIW
                b - natural parameter of NIW
                beta - parameter of NIW, is also called kappa
                v_hat - egree of freedom parameter of NIW
            nb_components: number of mixture components
        '''
        # make parameters for PGM part of recognition network
                
        mu_k, L_k = self.make_loc_scale_variables(theta, param_device)
        pi_k_softmax = torch.nn.Softmax(dim=0).cuda()
        #pi_k = pi_k_softmax(torch.randn((nb_components,)).to(param_device)) # does not go into module.parameters()


        return mu_k, L_k, pi_k_softmax


    def make_loc_scale_variables(self, theta, param_device):
        
        '''
        Args:
            theta [is a tuple that contains the following parameters]:
                alpha - the weights of the mixtures
                A - natural parameter of NIW
                b - natural parameter of NIW
                beta - parameter of NIW, is also called kappa
                v_hat - egree of freedom parameter of NIW
            param_device: location of where parameters are calculated
        '''
        theta_copied = niw.natural_to_standard(theta[1].clone(),theta[2].clone(),theta[3].clone(),theta[4].clone())

        mu_k_init, sigma_k = niw.expected_values(theta_copied)
        L_k_init = torch.cholesky(sigma_k)

        # mu_k = init_tensor_gpu_grad(mu_k_init,trainable=True,device=param_device)
        # L_k = init_tensor_gpu_grad(L_k_init,trainable=True,device=param_device)

        # For debugging seeing if this is causing the backwards problem:
        mu_k = init_tensor_gpu_grad(mu_k_init,trainable=False,device=param_device)
        L_k = init_tensor_gpu_grad(L_k_init,trainable=False,device=param_device)


        return mu_k, L_k


    #def forward(self, y, phi_gmm, encoder_layers, decoder_layers, nb_samples=10, stddev_init_nn=0.01, seed=0):
    def forward(self, y):

        # Assume currently MINST data set, where first index is data, second is labels, and data is sorted as Size x Image_Row x Image_Col
        # assert list(y.shape[-2:]) == [28, 28], "The INPUT is not MNIST"

        # Use VAE encoder
        # x_given_y_phi = self.x_given_y_phi_model.forward(y.view(-1, 784).to(self.device))
        x_given_y_phi = self.x_given_y_phi_model.forward(y.to(self.device))
        print ("Finished Encoder Forward pass at iteration {}".format(self.totiter))

        # execute E-step (update/sample local variables)
        x_k_samples, log_z_given_y_phi, phi_tilde, w_eta_12 = self.e_step(x_given_y_phi,  (self.train_mu_k, self.train_L_k, self.train_pi_k), self.nb_components, seed=0)
        print ("Finished E-step Forward pass at iteration: {}".format(self.totiter))
        # compute reconstruction
        
        y_reconstruction = self.y_reconstruction_model.forward(x_k_samples)
        print ("Finished Decoder Forward pass at iteration: {}".format(self.totiter))
        #temp = torch.tensor(0,dtype=torch.int64)
        x_samples = self.subsample_x(x_k_samples, log_z_given_y_phi, seed=0)[:, 0, :]

        return y_reconstruction, x_given_y_phi, x_k_samples, x_samples, log_z_given_y_phi,  (self.train_mu_k, self.train_L_k, self.train_pi_k), phi_tilde


    def e_step(self, phi_enc, phi_gmm, nb_samples, seed=0):
        """

        Args:
            phi_enc: encoded data; Base Measure Natural Parameters [In this case mean and variance for Gaussian]
            phi_gmm: parameters of the Graphical model, mu, variance, and the cluster weight
            nb_samples: number of ties to sample from q(x|z,y)
            seed: random seed

        Returns:

        """

        # Natural Parameter Vector of Encoder 
        # [see http://www.robots.ox.ac.uk/~cvrg/michaelmas2004/VariationalInferenceAndVMP.pdf slide 31]
        eta1_phi1, eta2_phi1_diag = phi_enc
        # diagonalize the percision/variance [shapes goes from (2,4) to (2,4,4)]
        eta2_phi1 = torch.diag_embed(eta2_phi1_diag)

        #unpack cluster weight and natural parameters
        eta1_phi2, eta2_phi2, pi_phi2 = self.unpack_recognition_gmm(phi_gmm)


        # compute log q(z|y, phi)
        log_z_given_y_phi, dbg = self.compute_log_z_given_y(eta1_phi1, eta2_phi1, eta1_phi2, eta2_phi2, pi_phi2)

        # compute parameters phi_tilde -- equations 20-24 in Wu Lin, Emtiyaz Khan Structured Inference Networks Paper
        # eta2_phi_tilde = eta2_phi1 + eta2_phi2
        # eta1_phi_tilde = inv(eta2_phi_tilde) * (eta1_phi1 + eta1_phi2)
        # eta1_phi_tilde.shape = (N, K, D, 1); eta2_phi_tilde.shape = (N, K, D, D)
        eta2_phi_tilde = eta2_phi1.unsqueeze(1) + eta2_phi2.unsqueeze(0) 
        eta1_phi_tilde = (eta1_phi1.unsqueeze(1) + eta1_phi2.unsqueeze(0)).unsqueeze(-1) # without inv(eta2_phi_tilde)

        x_k_samples = self.sample_x_per_comp(eta1_phi_tilde,eta2_phi_tilde,nb_samples,seed=0)

        return x_k_samples, log_z_given_y_phi, (eta1_phi_tilde, eta2_phi_tilde), dbg

        
    def sample_x_per_comp(self, eta1, eta2, nb_samples, seed=0):
        """
        Args:
            eta1: 1st Gaussian natural parameter, shape = N, K, L, 1
            eta2: 2nd Gaussian natural parameter, shape = N, K, L, L
            nb_samples: nb of samples to generate for each of the K components
            seed: random seed

        Returns:
            x ~ N(x|eta1[k], eta2[k]), nb_samples times for each of the K components.
        """

        inv_sigma = -2. * eta2  # For reason see e_step calculation of eta1_phi_tilde
        N, K, _, D = eta2.shape

        # cholesky decomposition and adding noise (raw_noise is of dimension (DxB), where B is the size of MC samples)
        # Note cholesky decomposition that the lower triangle can be interperted as the square root of the matrix
        L = torch.cholesky(inv_sigma) # sigma = sqrt(variance)
        sample_shape = (int(N), int(K), int(D), nb_samples)
        raw_noise = torch.randn(sample_shape).cuda()
        noise = L.transpose(dim0=3,dim1=2).inverse()@raw_noise

        # reparam-trick-sampling: x_samps = mu_tilde + noise: shape = N, K, S, D (permute = N-dim transpose)
        x_k_samps = (inv_sigma.inverse()@eta1 + noise).permute(0,1,3,2)

        return x_k_samps
        

    def subsample_x(self, x_k_samples, log_q_z_given_y, seed=0):
        """
        Given S samples for each of the K components for N datapoints (x_k_samples) and q(z_n=k|y), subsample S samples for
        each data point
        Args:
            x_k_samples: sample matrix of shape (N, K, S, L) 
            log_q_z_given_y: probability q(z_n=k|y_n, phi) [Shape: N x K]
            seed: random seed
        Returns:
            x_samples: a sample matrix of shape (N, S, L)
        """

        N, K, S, L = x_k_samples.shape

        # prepare indices for N and S dimension
        n_idx = torch.arange(start=0,end=N).unsqueeze(1).repeat(1,S)  # S samples for each observation N, n_idx[0] = len([0,0,0,...,0]) = S
        s_idx = torch.arange(start=0,end=S).unsqueeze(0).repeat(N,1)  # N Each observation has S samples, s_idx[0] = [0,1,2,...,S] -- N Times 

        tempvar = torch.exp(log_q_z_given_y.detach()).cpu()
        temp = tempvar.sum(dim=1)
        if (temp == 0).nonzero().nelement() != 0:
            print ("TEST zamps")
        z_samps = torch.multinomial(tempvar,S)
        
        # Make sure all indexes are ints
        z_samps = z_samps.to(torch.int64)

        # tensor of shape (N, S, 3), containing indices of all chosen samples
        # choices = torch.cat((n_idx.unsqueeze(2),z_samps.unsqueeze(2),s_idx.unsqueeze(2)),dim=2) --- DON'T NEED TO DO IN PYTORCH

        # select the chosen samples from x_k_samples, choices are the indices needed to extract from x_k_samples
        # So to paraphrase again, we have K components (from the GMM model) and S samples of each component, where each sample represents the parameters
        # of the latent dimensions, and what we want is S samples for each unique observation in the batch (N) such that the resulting matrix has
        # S samples of N observations

        # For example if we have a minibatch of 64, N = 64, GMM clusters of 10, K = 10, 10 Samples for every cluster, S = 10, and 3 Latent Dims from NN, then L = 6
        # So we have x_k_samples = [64, 10, 10, 6]  then n_idx represents getting S samples for observation, and s_idx represents which sample to get from the K-the component
        # the Kth-Component is chosen from z_samps, such that z_samps will have a length of S, so if z_samps = [9, 9, 9, 2, 9, 9, 9, 9, 8, 2], we will have 
        # 7 samples from the 9th component, 2 samples from the 2nd component, and 1 sample from the 8th component

        # Replaced tf.gather_nd in tensorflow (from original code) with pytorch's advance indexing
        
        return x_k_samples[[n_idx,z_samps,s_idx]]


    def unpack_recognition_gmm(self, phi_gmm):
        """
        
        Args:
            phi_gmm: Contains the parameters of the graphical model, specifically, natural parameters of mean and precision
            and the cluster weight
        
        Returns:
            Returns a tuple with the natural parameter of the mean and precision (1/variance) and the cluster weight
        
        """

        eta1, L_k_raw, pi_k_raw = phi_gmm
        temp = pi_k_raw.cpu().numpy()
        np.savetxt('test_1_{}.txt'.format(self.totiter),temp,fmt='%1.4e',delimiter=',')

        # Computer Precision - the inverted Variance (1/sigma^2)
        # Make sure L_k_raw is a valid Cholesky decomposition A = LL*, where L is lower triangle
        # L* is conjugate tranpose of L 
        L_k = torch.tril(L_k_raw) # Returns batch of lower triangular part of matrix

        # Get diagonals of lower triangular (Note for batch inputs need to do :To take a batch diagonal, pass in dim1=-2, dim2=-1)
        # see https://pytorch.org/docs/stable/torch.html#torch.diagonal
        diag_L_k = torch.diagonal(L_k,dim1=-2,dim2=-1)
        m = torch.nn.Softplus() # Softplus function to make sure everything is positive-definite
        softplus_L_k = m(diag_L_k)

        # Need to set diagonal of original Variance matrix to Softplus values, so use mask
        # see: https://stackoverflow.com/questions/49429147/replace-diagonal-elements-with-vector-in-pytorch/49431180#49431180
        mask = torch.diag_embed(torch.ones_like(softplus_L_k)) 
        L_k = torch.diag_embed(softplus_L_k) + (1. - mask)*L_k # * is overloaded in pytorch for elemente wise multiplication

        # Compute Precision [Note @ = matmul]  
        the_Precision = L_k @ torch.transpose(L_k,2,1) #dim's 1 and 2 are the cluster parameters, dim 0 are the actual clusters

        # Compute natural parameter of precision 
        eta2 = -0.5*the_Precision

        # make sure that log_pi_k are valid mixture coefficients, softmax normalizes pi_k such that sum(pi_k)=1
        m = torch.nn.Softmax(dim=0) 
        pi_k = m(pi_k_raw)

        return (eta1, eta2, pi_k)

    def compute_log_z_given_y(self, eta1_phi1, eta2_phi1, eta1_phi2, eta2_phi2, pi_phi2):
        """

        Args:
            eta1_phi1: encoder output; shape = N, K, L
            eta2_phi1: encoder output; shape = N, K, L, L
            eta1_phi2: GMM-EM parameter; shape = K, L
            eta2_phi2: GMM-EM parameter; shape = K, L, L
            where N = batch size, K = Number of Clusters, L = Number of Latent Variables

        Returns:
            log q(z|y, phi)
        """

        N, L = eta1_phi1.shape # mean * precision
        assert list(eta2_phi1.shape) == [N, L, L]
        K, L2 = eta1_phi2.shape # mean * precision
        assert L2 == L
        assert list(eta2_phi2.shape) == [K, L, L] # 1/precision

        # Get Natural Parameters of Gaussian -- again see: http://www.robots.ox.ac.uk/~cvrg/michaelmas2004/VariationalInferenceAndVMP.pdf (slide 31)
        # eta2 = precision * -0.5
        # eta1 = mean * precision
        # where precision = inverse(variance)

        # z ~ N(mu_phi1|mu_phi2,sigma_phi1+sigma+phi2) [Lin, Kahn, VMP + SVAE pg. 6]
        # so percision is inv(sigma_ph1+sigma_phi2)
        # since we have natural parameters eta2_1 and eta2_2, we need to calculate natural parameter
        # inv(sigma_phi1+sigma_phi2) from eta2_1 and eta2_2 which is
        # [eta2_1*eta2_2] / [eta2_1 + eta2_2] = inverse(sigma_phi1 + sigma_phi2)


        # combine eta2_phi1 and eta2_phi2 - eta2_phi1 has dimensions mini-batch samples x latent x latent and eta2_phi2 has dimensions num_components x latent dim x latent dim
        # output is now mini_batch x num_components x latent_dim x latent_dim
        eta2_phi_tilde = eta2_phi1.unsqueeze(1) + eta2_phi2.unsqueeze(0)
        
        # calculate eta2_2 / inverse(eta2_2 + eta2_1) = inverse(eta2_2+eta2_1) * eta2_2 [shape:  mini_batch x num_components x latent_dim x latent_dim]
        inv_eta2_eta2_sum_eta1 = eta2_phi_tilde.inverse() @ eta2_phi2.expand(N,-1,-1,-1)

        # calculate eta2_1 * inv_sum_eta2_eta1 [shape:  mini_batch x num_components x latent_dim x latent_dim]
        w_eta2 = torch.einsum('nju,nkui->nkij', eta2_phi1, inv_eta2_eta2_sum_eta1)

        # now calculate the mean natural parameter [mean * precision]
        # remember precision is inv[sigma_phi1+sigma+phi2]

        # calculate [mu*precision_2] * (1 / eta2_2 + eta2_1)  --- Note eta1_phi2 = mu*precision_2 or mu/sigma_phi2
        mu_eta2_1_eta2_2 = eta2_phi_tilde.inverse() @ eta1_phi2.unsqueeze(0).unsqueeze(-1).expand(N,-1,-1,1) # Shape: NxKxLx1

        #Calculate eta2_1 * mu_eta2_1_eta2_2 = [mu*eta2_1*eta2_2]/(eta2_1+eta2_2)
        w_eta1 = torch.einsum('nuj,nkuv->nkj',eta2_phi1,mu_eta2_1_eta2_2) # Shape: NxKxL

        # compute means
        mu_phi1, _ = gaussian.natural_to_standard(eta1_phi1, eta2_phi1)  # Remember the observed data are the means of recognition network (encoder output)

        # computer log_z_given_y_phi
        return gaussian.log_probability_nat(mu_phi1, w_eta1, w_eta2, pi_phi2), (w_eta1, w_eta2)


    def compute_elbo(self, y, reconstructions, theta, phi_tilde, x_k_samps, log_z_given_y_phi, decoder_type):
        """
        Compute ELBO of Latent GMM 
        Args:
            y: original data
            reconstructions: reconststructed y
            theta: hyperparameters of GMM model 
                [alpha: prior Dirichlet parameters, beta/kappa: prior NiW, 
                controls variance of mean, m: prior of mean, c: prior of covariance, v: prior degrees of freedom ]
            phi_tilde: Natural Parameters of GMM
            x_k_samps: Latent Vectors produced from GMM
            log_z_given_y_phi: Mixture probabilities # Shape: N x K 
            decoder_type: Gaussian or Bernoulli decoder

        Returns:
            ELBO: evidence lower bound of reconstruction and KL divergence of prior and variational prior
            Details: Tuple of negative reconstruction error, numberator of regularizer, denominator of regulaizer, regualizer term
        """
        
        beta_k, m_k, C_k, v_k = niw.natural_to_standard(*theta[1:])
        mu, sigma = niw.expected_values((beta_k, m_k, C_k, v_k))
        eta1_theta, eta2_theta = gaussian.standard_to_natural(mu, sigma)
        alpha_k = dirichlet.natural_to_standard(theta[0])
        expected_log_pi_theta = dirichlet.expected_log_pi(alpha_k)

        # Don't backprop through GMM
        eta1_theta = eta1_theta.detach()  
        eta2_theta = eta2_theta.detach()
        expected_log_pi_theta = expected_log_pi_theta.detach()

        r_nk = torch.exp(log_z_given_y_phi)

        # compute negative reconstruction error; sum over minibatch (use VAE function)
        means_recon, out_2_recon = reconstructions # out_2 is gaussian variances 
        if decoder_type == 'standard':
            self.neg_reconstruction_error = self.expected_diagonal_gaussian_loglike(y.view(-1, 784).to(self.device), means_recon, out_2_recon, weights=r_nk)
        else:
            raise NotImplementedError
        
        # compute E[log q_phi(x,z=k|y)]
        eta1_phi_tilde, eta2_phi_tilde = phi_tilde
        N, K, L, _ = eta2_phi_tilde.shape
        eta1_phi_tilde = torch.reshape(eta1_phi_tilde, (N, K, L))

        N, K, S, L = x_k_samps.shape

        # Computer Log-Numerator see: Variational Message Parsing with Structured Inference Networks pg. 5 Equations 7 - 10
        # Log-Numerator = log[ PROD(p(y_n|x_n, theta_NN) * p(x|theta_PGM) * Z(phi)]
        # Log-Denominator = log[ PROD(q(x_n|f_phi_nn(y_n)*q(x|phi_PGM))]  
        
        # Note p(y_n|x_n, theta_NN) / q(x_n|f_phi_nn(y_n) is the RECONSTRUCTION ERROR   
        # The unique parts of this ELBO are E_q[log p(x|theta_PGM)] - E_q[log q(x|phi_PGM)] - log Z(phi)

        # For GMM Z(phi) = sum_{1 to K}(N(m_n|mean_tilde_k, V_n+sigma_tilde_k) * pi_k
        # where m_n = mean of encoder, V_n = variance of encoder
        # sigma_tilde_n = inverse(V_n) + inverse(sigma_tilde_k)
        # mean_tilde_n = sigma_tilde_n * (inverse(V_n)*m_n + inverse(sigma_tilde_k)*mean_tilde_k)
        # This results in Z(phi) = sum_{1 to K}(log_z_given_y_phi)

        # Log Numerator = q(x,z|y) = q(x|z, y, phi)*q(z|y,phi) = N(x_n|mean_tilde_n,sigma_tilde_n)*N(m_n|mean_tild_k,V_n+sigma_tilde_k)
        log_N_x_given_phi = gaussian.log_probability_nat_per_samp(x_k_samps, eta1_phi_tilde, eta2_phi_tilde)  # Shape: N x K x L  
        log_numerator = log_N_x_given_phi + log_z_given_y_phi.unsqueeze(2)  # Since q(z|y,phi) is only of shape N x K

        log_N_x_given_theta = gaussian.log_probability_nat_per_samp(x_k_samps, eta1_theta.unsqueeze(0).expand(N,-1,-1), eta2_theta.expand(N,-1,-1,-1)) # Shape: N x K x L
        log_denominator = log_N_x_given_theta + expected_log_pi_theta.unsqueeze(0).unsqueeze(2)

        regualizer_term_part_1 = r_nk.unsqueeze(2) * (log_numerator - log_denominator)
        regualizer_term_part_2 = torch.sum(regualizer_term_part_1,dim=1)
        regualizer_term_part_3 = torch.sum(regualizer_term_part_2,dim=0)
        self.regualizer_term_final = torch.mean(regualizer_term_part_3)

        elbo = -1. * (self.neg_reconstruction_error - self.regualizer_term_final)

        details = (self.neg_reconstruction_error, torch.sum(r_nk*torch.mean(log_numerator,-1)),torch.sum(r_nk*torch.mean(log_denominator,-1)), self.regualizer_term_final)
        self.totiter += 1

        return elbo, details
    
    def compute_elbo_debug(self, y, reconstructions, theta, phi_tilde, x_k_samps, log_z_given_y_phi, decoder_type):
        """
        Compute Reconstruction Error  -- For debugging purposes
        Args:
            y: original data
            reconstructions: reconststructed y
            theta: hyperparameters of GMM model 
                [alpha: prior Dirichlet parameters, beta/kappa: prior NiW, 
                controls variance of mean, m: prior of mean, c: prior of covariance, v: prior degrees of freedom ]
            phi_tilde: Natural Parameters of GMM
            x_k_samps: Latent Vectors produced from GMM
            log_z_given_y_phi: Mixture probabilities # Shape: N x K 
            decoder_type: Gaussian or Bernoulli decoder

        Returns:
            ELBO: evidence lower bound of reconstruction and KL divergence of prior and variational prior
            Details: Tuple of negative reconstruction error, numberator of regularizer, denominator of regulaizer, regualizer term
        """

        # Don't backprop through GMM
        r_nk = torch.exp(log_z_given_y_phi)

        # compute negative reconstruction error; sum over minibatch (use VAE function)
        means_recon, out_2_recon = reconstructions # out_2 is gaussian variances 
        if decoder_type == 'standard':
            self.neg_reconstruction_error = self.expected_diagonal_gaussian_loglike(y.to(self.device), means_recon, out_2_recon, weights=r_nk)
        else:
            raise NotImplementedError

        elbo = -1. * (self.neg_reconstruction_error)
        self.totiter += 1

        return elbo


    def expected_diagonal_gaussian_loglike(self, y, param1_recon, param2_recon, weights=None):
        """
        computes expected diagonal log-likelihood SUM_{n=1} E_{q(z)}[log N(x_n|mu(z), sigma(z))]
        Args:
            y: data
            param1_recon: predicted means; shape (size_minibatch, nb_samples, dims) or (size_minimbatch, nb_comps, nb_samps, dims)
            param2_recon: predicted variances; shape is same as for means
            weights: None or matrix of shape (N, K) containing normalized weights

        Returns:

        """

        if weights is None:
            # required dimension: size_minibatch, nb_samples, dims
                        
            param1_recon = param1_recon if len(param1_recon.shape) == 3 else param1_recon.unsqueeze(1)
            param2_recon = param2_recon if len(param2_recon.shape) == 3 else param2_recon.unsqueeze(1)
            M, S, L = param1_recon.shape
            assert list(y.shape) == [M, L]

            sample_mean = torch.sum(torch.pow(y.unsqueeze(1) - param1_recon, 2) / param2_recon) + torch.sum(torch.log(param2_recon))

            S = torch.tensor(int(S), dtype=torch.float32, requires_grad = False)
            M = torch.tensor(int(M), dtype=torch.float32, requires_grad = False)
            L = torch.tensor(int(L), dtype=torch.float32, requires_grad = False)
            pi = torch.tensor(np.pi, dtype=torch.float32, requires_grad = False)

            sample_mean /= S
            loglik = -1/2 * sample_mean - M * L/2. * torch.log(2. * pi)
        
        else:
            M, K, S, L = param1_recon.shape
            assert param2_recon.shape == param1_recon.shape
            assert list(weights.shape) == [M, K]

            # adjust y's shape (add component and sample dimensions)
            y = y.unsqueeze(1).unsqueeze(1)

            sample_mean = torch.einsum('nksd,nk->', torch.pow(y - param1_recon,2)/ param2_recon + torch.log(param2_recon + 1e-8), weights)

            S = torch.tensor(int(S), dtype=torch.float32, requires_grad = False).cuda()
            M = torch.tensor(int(M), dtype=torch.float32, requires_grad = False).cuda()
            L = torch.tensor(int(L), dtype=torch.float32, requires_grad = False).cuda()
            pi = torch.tensor(np.pi, dtype=torch.float32, requires_grad = False).cuda()

            sample_mean /= S
            loglik = -1/2 * sample_mean - M * L/2. * torch.log(2. * pi)
        
        return loglik


    def update_gmm_params(self, current_gmm_params, gmm_params_star, step_size):
        """
        Computes convex combination between current and updated parameters.
        Args:
            current_gmm_params: current parameters
            gmm_params_star: parameters received by GMM-EM algorithm
            step_size: step size for convex combination
            name:

        Returns:
        """
        a, b, c, d, e = current_gmm_params
        step_size = torch.from_numpy(np.array(step_size)).cuda()

        current_gmm_params = [(1 - step_size)*curr_param + step_size * param_star for (curr_param, param_star) in zip(current_gmm_params, gmm_params_star)]
  
        return current_gmm_params


    def predict(self, y, phi_gmm, encoder_layers, decoder_layers, seed=0):
        """
        Args:
            y: data to cluster and reconstruct
            phi_gmm: latent phi param
            encoder_layers: encoder NN architecture
            decoder_layers: encoder NN architecture
            seed: random seed

        Returns:
            reconstructed y and most probable cluster allocation
        """

        nb_samples = 1
        phi_enc_model = Encoder(layerspecs=encoder_layers)
        phi_enc = phi_enc_model.forward(y)

        x_k_samples, log_r_nk, _, _ = e_step(phi_enc, phi_gmm, nb_samples, seed=0)
        x_samples = subsample_x(x_k_samples, log_r_nk, seed)[:, 0, :]

        y_recon_model = Decoder(layerspecs=decoder_layers)
        y_mean, _ = y_recon_model.forward(x_samples)

        return (y_mean, torch.argmax(log_r_nk,dim=1))


    def m_step(self, gmm_prior, x_samples, r_nk):
        """
        Args:
            gmm_prior: Dirichlet+NiW prior for Gaussian mixture model
            x_samples: samples of shape (N, S, L)
            r_nk: responsibilities of shape (N, K)

        Returns:
            Dirichlet+NiW parameters obtained by executing Bishop's M-step in the VEM algorithm for GMMs
        """


        # execute GMM-EM m-step
        beta_0, m_0, C_0, v_0 = niw.natural_to_standard(*gmm_prior[1:])
        alpha_0 = dirichlet.natural_to_standard(gmm_prior[0])

        alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k = gmm.m_step(x_samples, r_nk, alpha_0, beta_0, m_0, C_0, v_0)

        A, b, beta, v_hat = niw.standard_to_natural(beta_k, m_k, C_k, v_k)
        alpha = dirichlet.standard_to_natural(alpha_k)

        return (alpha, A, b, beta, v_hat)

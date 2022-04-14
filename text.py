from __future__ import division
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import os,sys
import time
import dateutil.tz
import datetime
import argparse
import importlib
import random
import torch.optim as optim
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

class GMM_sampler(object):
    def __init__(self, N, mean=None, n_components=None, cov=None, sd=None, dim=None, weights=None):
        np.random.seed(1024)
        self.total_size = N
        self.n_components = n_components
        self.dim = dim
        self.sd = sd
        self.weights = weights
        if mean is None:
            assert n_components is not None and dim is not None and sd is not None
            self.mean = np.random.uniform(-5,5,(self.n_components,self.dim))
        else:
            assert cov is not None    
            self.mean = mean
            self.n_components = self.mean.shape[0]
            self.dim = self.mean.shape[1]
            self.cov = cov
        if weights is None:
            self.weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=N, replace=True, p=self.weights)
        if mean is None:
            self.X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in self.Y],dtype='float64')
        else:
            self.X = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        self.X_train, self.X_val,self.X_test = self.split(self.X)

class GMM_indep_sampler(object):
    def __init__(self, N, sd, dim, n_components, weights=None, bound=1):
        np.random.seed(1024)
        self.total_size = N
        self.dim = dim
        self.sd = sd
        self.n_components = n_components
        self.bound = bound
        self.centers = np.linspace(-bound, bound, n_components)
        self.X = np.vstack([self.generate_gmm() for _ in range(dim)]).T
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.nb_train = self.X_train.shape[0]
        self.Y=None
    def generate_gmm(self,weights = None):
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        return np.array([np.random.normal(self.centers[i],self.sd) for i in Y],dtype='float64')
    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
    
    def get_density(self, data):
        assert data.shape[1]==self.dim
        from scipy.stats import norm
        centers = np.linspace(-self.bound, self.bound, self.n_components)
        prob = []
        for i in range(self.dim):
            p_mat = np.zeros((self.n_components,len(data)))
            for j in range(len(data)):
                for k in range(self.n_components):
                    p_mat[k,j] = norm.pdf(data[j,i], loc=centers[k], scale=self.sd) 
            prob.append(np.mean(p_mat,axis=0))
        prob = np.stack(prob)        
        return np.prod(prob, axis=0)

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y

class UCI_sampler(object):
    def __init__(self,data_path='datasets/data.npy'):
        data = np.load(data_path)
        self.X_train, self.X_val, self.X_test = self.normalize(data)
        self.Y = None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def normalize(self,data):
        rng = np.random.RandomState(42)
        rng.shuffle(data)
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None
    
class Gaussian_sampler(object):
    def __init__(self, mean, sd=1, N=10000):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean)))

    def load_all(self):
        return self.X, self.Y
    
class DataPool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.nb_batch = 0
        self.pool = []

    def __call__(self, data):
        if self.nb_batch < self.maxsize:
            self.pool.append(data)
            self.nb_batch += 1
            return data
        if np.random.rand() > 0.5:
            results=[]
            for i in range(len(data)):
                idx = int(np.random.rand()*self.maxsize)
                results.append(copy.copy(self.pool[idx])[i])
                self.pool[idx][i] = data[i]
            return results
        else:
            return data

class Discriminator(nn.Module):
    def __init__(self, input_dim, nb_layers=4, nb_units=512):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        if (type(input_dim) is tuple and len(input_dim) > 1):
            inp = int(np.prod(input_dim))
            self.tag = 1
        else:
            self.tag = 0
            inp = int(input_dim)

        modules = nn.Sequential()
        modules.add_module('linear', nn.Linear(inp, nb_units))
        modules.add_module('leakyRelu', nn.LeakyReLU())
        for i in range(nb_layers):
            modules.add_module('linear_{}'.format(i), nn.Linear(nb_units, nb_units))
            modules.add_module('tanh_{}'.format(i), nn.Tanh())

        modules.add_module('linear_{}'.format(nb_layers+1), nn.Linear(nb_units, 1))
        self.model = modules

    def forward(self, img):
        if self.tag:
            img = img.view(img.size(0), -1)
        validity = self.model(img)
        return validity

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, nb_layers=4, nb_units=512):
        super(Generator, self).__init__()
        self.output_dim = output_dim

        if (type(output_dim) is tuple and len(output_dim) > 1):
            out = int(np.prod(output_dim))
            self.tag = 1
        else:
            self.tag = 0
            out = int(output_dim)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU())
            return layers

        modules = nn.Sequential(*block(input_dim, nb_units, normalize=False))

        for i in range(nb_layers):
            modules.add_module('linear_{}'.format(i), nn.Linear(nb_units, nb_units))
            modules.add_module('leakyRel_{}'.format(i), nn.LeakyReLU())

        modules.add_module('linear_{}'.format(nb_layers+1), nn.Linear(nb_units, out))
        self.model = modules

    def forward(self, z):
        out = self.model(z)
        if self.tag:
            out = out.view(out.size(0), *self.out_shape)
        return out

class RoundtripModel(object):
    def __init__(self, g_net, h_net, dx_net, dy_net, x_sampler, y_sampler, data, pool, batch_size, alpha, beta, df, is_train):
        self.device = "cuda"
        self.g_net = g_net.to(self.device)
        self.h_net = h_net.to(self.device)
        self.dx_net = dx_net.to(self.device)
        self.dy_net = dy_net.to(self.device)
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.df = df
        self.data = data
        self.pool = pool
        self.x_dim = self.dx_net.input_dim
        self.y_dim = self.dy_net.input_dim
        self.lr = 2e-4
        
        self.g_h_optim = torch.optim.Adam(list(self.g_net.parameters()) + list(self.h_net.parameters()), lr = self.lr, betas = (0.5, 0.9))
        self.d_optim = torch.optim.Adam(list(self.dx_net.parameters()) + list(self.dy_net.parameters()), lr = self.lr, betas = (0.5, 0.9))
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        #save path for saving predicted data
        self.save_dir = 'data/density_est_{}_{}_x_dim={}_y_dim={}_alpha={}_beta={}'.format(self.timestamp,self.data,self.x_dim, self.y_dim, self.alpha, self.beta)
        if not os.path.exists(self.save_dir) and is_train:
            os.makedirs(self.save_dir)
        
    def generator_loss(self, x, y):
        y_ = self.g_net(x)
        x_ = self.h_net(y)

 #       self.J = batch_jacobian(y_, x)
        x__ = self.h_net(y_)
        y__ = self.g_net(x_)

        dy_ = self.dy_net(y_)
        dx_ = self.dx_net(x_)

        l1_loss_x = torch.mean(torch.abs(x - x__))
        l1_loss_y = torch.mean(torch.abs(y - y__))

        l2_loss_x = torch.mean((x - x__)**2)
        l2_loss_y = torch.mean((y - y__)**2)
        
        g_loss_adv = torch.mean((0.9*torch.ones_like(dy_) - dy_)**2)
        h_loss_adv = torch.mean((0.9*torch.ones_like(dx_) - dx_)**2)
        
        g_loss = g_loss_adv + self.alpha*l2_loss_x + self.beta*l2_loss_y
        h_loss = h_loss_adv + self.alpha*l2_loss_x + self.beta*l2_loss_y
        g_h_loss = g_loss_adv + h_loss_adv + self.alpha*l2_loss_x + self.beta*l2_loss_y
        
        return g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_y, g_loss, h_loss, g_h_loss
        
    def discriminator_loss(self, x, y):
        fake_y = self.g_net(x)
        fake_x = self.h_net(y)
        
        dx = self.dx_net(x)
        dy = self.dy_net(y)

        d_fake_x = self.dx_net(fake_x)
        d_fake_y = self.dy_net(fake_y)

        dx_loss = (torch.mean((0.9*torch.ones_like(dx) - dx)**2)+torch.mean((0.1*torch.ones_like(d_fake_x) - d_fake_x)**2))/2.0
        dy_loss = (torch.mean((0.9*torch.ones_like(dy) - dy)**2)+torch.mean((0.1*torch.ones_like(d_fake_y) - d_fake_y)**2))/2.0
        d_loss = dx_loss + dy_loss
        return fake_x, fake_y, dx_loss, dy_loss, d_loss


    def train(self, epochs, cv_epoch, patience):
        data_y_train = copy.copy(self.y_sampler.X_train)
        data_y_test = self.y_sampler.X_test
        data_y_val = self.y_sampler.X_val
        best_likelihood_val = -np.inf
        counter = 1
        start_time = time.time()
        for epoch in range(epochs):
            np.random.shuffle(data_y_train)
            batch_idxs = len(data_y_train) // self.batch_size
            for idx in range(batch_idxs):
                bx = self.x_sampler.get_batch(self.batch_size)
                by = data_y_train[self.batch_size*idx:self.batch_size*(idx+1)]
                
                #quick test on a random batch data
                if counter % 100 == 0:
                    bx = self.x_sampler.train(batch_size)
                    by = self.y_sampler.train(batch_size)
                    x = torch.Tensor(bx).to(self.device)
                    y = torch.Tensor(by).to(self.device)
                    g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_y, g_loss, h_loss, g_h_loss = self.generator_loss(x, y)
                    fake_bx, fake_by, dx_loss, dy_loss, d_loss = self.discriminator_loss(x, y)
                    print('Epoch [%d] Iter [%d] Time [%5.4f] g_loss_adv [%.4f] h_loss_adv [%.4f] l2_loss_x [%.4f] \
                        l2_loss_y [%.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] \
                        dy_loss [%.4f] d_loss [%.4f]' %
                        (epoch, counter, time.time() - start_time, g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_y, \
                        g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss))                 
                counter+=1
            if epoch == cv_epoch:
                global best_sd, best_scale
                if use_cv:
                    best_sd, best_scale = self.model_selection()
                f_val = open('%s/log_val.txt'%self.save_dir,'a+')
                f_test = open('%s/log_test.txt'%self.save_dir,'a+')
                f_val.write('epoch\taverage_likelihood\tstandard_deviation\n')
                f_test.write('epoch\taverage_likelihood\tstandard_deviation\n')
            if epoch >= cv_epoch:
                self.save()
                wait = 0
                py_est_val = self.estimate_py_with_IS(data_y_val,epoch,sd_y=best_sd,scale=best_scale,sample_size=2000,log=True,save=False)
                average_likelihood_val = np.mean(py_est_val)
                sd_likelihood_val = np.std(py_est_val)/np.sqrt(len(py_est_val))
                f_val.write('%d\t%f\t%f\n'%(epoch,average_likelihood_val,sd_likelihood_val))
                if average_likelihood_val > best_likelihood_val:
                    best_likelihood_val = average_likelihood_val
                    py_est_test = self.estimate_py_with_IS(data_y_test,epoch,sd_y=best_sd,scale=best_scale,sample_size=2000,log=True)
                    average_likelihood_test = np.mean(py_est_test)
                    sd_likelihood_test = np.std(py_est_test)/np.sqrt(len(py_est_test))
                    f_test.write('%d\t%f\t%f\n'%(epoch,average_likelihood_test,sd_likelihood_test))
                else:
                    wait +=1
                    if wait>patience or epoch+1==epochs:
                        print('Early stopping at %d with best sd:%f, best scale:%f, test average likelihood%f, test sd likelihood%f'%(epoch,best_sd,best_scale, average_likelihood_test,sd_likelihood_test))
                        f_val.close()
                        f_test.close()
                        sys.exit()
            
    #selection the best sd and scale 
    def model_selection(self,sample_size=20000):
        data_y_val = self.y_sampler.X_val
        sd_list = [0.05,0.1,0.5]
        scale_list = [0.005,0.01,0.1,0.5,1]
        records = []
        for sd in sd_list:
            for scale in scale_list:
                py_est = self.estimate_py_with_IS(data_y_val,0,sd_y=sd,scale=scale,sample_size=sample_size,log=True,save=False)
                records.append([sd,scale,np.mean(py_est)])
        #sort according to the likelihood of validation set
        records.sort(key=lambda item:item[-1])
        best_sd, best_scale = records[-1][0],records[-1][1]
        return best_sd, best_scale

    #predict with y_=G(x)
    def predict_y(self, x, bs=256):
        assert x.shape[-1] == self.x_dim
        N = x.shape[0]
        y_pred = np.zeros(shape=(N, self.y_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
                ind = np.arange(b*bs, N)
            else:
                ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_x = torch.Tensor(batch_x).to(self.device)
            batch_y_ = self.g_net(batch_x)
            y_pred[ind, :] = batch_y_.detach().cpu().numpy()
        return y_pred
    
    #predict with x_=H(y)
    def predict_x(self,y,bs=256):
        assert y.shape[-1] == self.y_dim
        N = y.shape[0]
        x_pred = np.zeros(shape=(N, self.x_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
                ind = np.arange(b*bs, N)
            else:
                ind = np.arange(b*bs, (b+1)*bs)
            batch_y = y[ind, :]
            batch_y = torch.Tensor(batch_y).to(self.device)
            batch_x_ = self.h_net(batch_y)
            x_pred[ind, :] = batch_x_.detach().cpu().numpy()
        return x_pred

    #calculate Jacobian matrix 
    def get_jacobian(self,x,bs=16):
        N = x.shape[0]
        jcob_pred = np.zeros(shape=(N, self.y_dim, self.x_dim)) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
                ind = np.arange(b*bs, N)
            else:
                ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_J = batch_jacobian(x, batch_x)
            jcob_pred[ind, :] = batch_J
        return jcob_pred

    #estimate pdf of y (e.g., p(y)) with importance sampling
    def estimate_py_with_IS(self,y_points,epoch,sd_y=0.5,scale=0.5,sample_size=40000,bs=1024,log=True,save=True):
        np.random.seed(0)
        from scipy.stats import t
        from multiprocessing.dummy import Pool as ThreadPool
        #multi-process to parallel the program
        def program_paral(func,param_list):
            pool = ThreadPool()
            results = pool.map(func,param_list)
            pool.close()
            pool.join()
            return results

        def py_given_x(zip_list):
            '''
            calculate p(y|x)
            x_points with shape (sample_size, x_dim)
            y_point wish shape (y_dim, )
            '''
            x_points = zip_list[0]
            y_point = zip_list[1]
            y_points_ = self.predict_y(x_points,bs=bs)
            if log:
                return -self.y_dim*np.log((np.sqrt(2*np.pi)*sd_y))-(np.sum((y_point-y_points_)**2,axis=1))/(2.*sd_y**2)
            else:
                return 1. / ((np.sqrt(2*np.pi)*sd_y)**self.y_dim) * np.exp(-(np.sum((y_point-y_points_)**2,axis=1))/(2.*sd_y**2))

        def w_likelihood_ratio(zip_list): #No Problem
            '''
            calculate w=px/py
            x_point with shape (x_dim, )
            x_points with shape (sample_size,x_dim)
            '''
            x_point = zip_list[0]
            x_points = zip_list[1]
            if log:
                log_qx = np.sum(t.logpdf(x_point-x_points,self.df,loc=0,scale=scale),axis=1)
                log_px = -self.x_dim*np.log(np.sqrt(2*np.pi))-(np.sum((x_points)**2,axis=1))/2.
                return log_px-log_qx
            else:
                qx =np.prod(t.pdf(x_point-x_points,self,loc=0,scale=scale),axis=1)
                px = 1. / (np.sqrt(2*np.pi)**self.x_dim) * np.exp(-(np.sum((x_points)**2,axis=1))/2.)
                return px / qx

        #sample a set of points given each x_point from importance distribution
        def sample_from_qx(x_point):
            '''
            multivariate student t distribution can be constructed from a multivariate Gaussian 
            one can also use t.rvs to sample (see the uncommented line) which is lower
            '''
            S = np.diag(scale**2 * np.ones(self.x_dim))
            z1 = np.random.chisquare(self.df, sample_size)/self.df
            z2 = np.random.multivariate_normal(np.zeros(self.x_dim),S,(sample_size,))
            return x_point + z2/np.sqrt(z1)[:,None]
            #return np.hstack([t.rvs(self.df, loc=value, scale=scale, size=(sample_size,1), random_state=None) for value in x_point])
        x_points_ = self.predict_x(y_points,bs=bs)
        N = len(y_points)
        py_given_x_list=[]
        w_likelihood_ratio_list=[]
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
                ind = np.arange(b*bs, N)
            else:
                ind = np.arange(b*bs, (b+1)*bs)
            batch_y_points = y_points[ind, :]
            batch_x_points_ = x_points_[ind, :]
            batch_x_points_sample_list = program_paral(sample_from_qx,batch_x_points_)
            batch_py_given_x_list = program_paral(py_given_x, zip(batch_x_points_sample_list, batch_y_points))
            batch_w_likelihood_ratio_list = program_paral(w_likelihood_ratio, zip(batch_x_points_, batch_x_points_sample_list))
            py_given_x_list += batch_py_given_x_list
            w_likelihood_ratio_list += batch_w_likelihood_ratio_list
        #calculate p(y)=int(p(y|x)*p(x)dx)=int(p(y|x)*w(x)q(x)dx)=E(p(y|x)*w(x)) where x~q(x)
        if log:
            #py_list = py_given_x_list + w_likelihood_ratio_list
            py_list = list(map(lambda x, y, : x + y, py_given_x_list, w_likelihood_ratio_list))
            max_idx_list = [np.where(item==max(item))[0][0] for item in py_list]
            py_est = np.array([np.log(np.sum(np.exp(item[0]-item[0][item[1]])))-np.log(sample_size)+item[0][item[1]] for item in zip(list(py_list),max_idx_list)])
        else:
            #py_list = py_given_x_list*w_likelihood_ratio_list
            py_list = list(map(lambda x, y, : x * y, py_given_x_list, w_likelihood_ratio_list))
            py_est = np.array([np.mean(item) for item in py_list])
        if save:
            np.save('%s/py_est_at_epoch%d.npy'%(self.save_dir,epoch), py_est)
        return py_est

    def save(self):
        if not os.path.exists('/model_saved/'):
            os.makedirs('/model_saved/')
        torch.save(self.g_net.state_dict(),
                   '/model_saved/g_net_{}'.format(self.data))
        torch.save(self.h_net.state_dict(),
                   '/model_saved/h_net_{}'.format(self.data))
        torch.save(self.dx_net.state_dict(),
                   '/model_saved/dx_net_{}'.format(self.data))
        torch.save(self.dy_net.state_dict(),
                   '/model_saved/dy_net_{}'.format(self.data))

    def load(self, pre_trained = False, timestamp='',epoch=999):
        self.g_net.load_state_dict(torch.load('/model_saved/g_net_{}'.format(self.data)))
        self.h_net.load_state_dict(torch.load('/model_saved/h_net_{}'.format(self.data)))
        self.dx_net.load_state_dict(torch.load('/model_saved/dx_net_{}'.format(self.data)))
        self.dy_net.load_state_dict(torch.load('/model_saved/dy_net_{}'.format(self.data)))
        print('Restored model weights.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='indep_gmm',help='name of data type')
#    parser.add_argument('--model', type=str, default='model',help='model path')
    parser.add_argument('--dx', type=int, default=10,help='dimension of latent space')
    parser.add_argument('--dy', type=int, default=10,help='dimension of data space')
    parser.add_argument('--bs', type=int, default=64,help='batch size for training')
    parser.add_argument('--ss', type=int, default=40000,help='sample size of importance sampling (IS)')
    parser.add_argument('--epochs', type=int, default=2000,help='maximum training epoches')
    parser.add_argument('--cv_epoch', type=int, default=20,help='epoch starting for evaluating')
    parser.add_argument('--patience', type=int, default=5,help='patience for early stopping')
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--use_cv', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=False)
#    parser.add_argument('--cuda', type=int, default=1, help='usage of cuda GPU')
    parser.add_argument('--df', type=float, default=1,help='degree of freedom of student t distribution')
    args = parser.parse_args()
    data = args.data
#    model = importlib.import_module(args.model)
    x_dim = args.dx
    y_dim = args.dy
    sample_size = args.ss
    batch_size = args.bs
    epochs = args.epochs
    cv_epoch = args.cv_epoch
    patience = args.patience
    alpha = args.alpha
    beta = args.beta
    df = args.df
    timestamp = args.timestamp
    use_cv = args.use_cv
    is_train = args.train
#    device = torch.device("cuda:0" if (cuda and torch.cuda.is_available()) else "cpu")
    
    g_net = Generator(input_dim=x_dim, output_dim = y_dim, nb_layers=10, nb_units=512)
    h_net = Generator(input_dim=y_dim, output_dim = x_dim, nb_layers=10, nb_units=256)
    dx_net = Discriminator(input_dim=x_dim, nb_layers=2, nb_units=128)
    dy_net = Discriminator(input_dim=y_dim, nb_layers=4, nb_units=256)
    pool = DataPool()

    xs = Gaussian_sampler(mean=np.zeros(x_dim),sd=1.0)

    if data == "indep_gmm":
        if not use_cv:
            best_sd, best_scale = 0.05, 0.5
        ys = GMM_indep_sampler(N=20000, sd=0.1, dim=y_dim, n_components=3, bound=1)
    else:
        print("Wrong data name!")
        sys.exit()

    RTM = RoundtripModel(g_net, h_net, dx_net, dy_net, xs, ys, data, pool, batch_size, alpha, beta, df, is_train)

    if args.train:
        RTM.train(epochs=epochs,cv_epoch=cv_epoch,patience=patience)
    else:
        print('Attempting to Restore Model ...')
        if timestamp == '':
            RTM.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            RTM.load(pre_trained=False, timestamp = timestamp, epoch = epochs-1)

          

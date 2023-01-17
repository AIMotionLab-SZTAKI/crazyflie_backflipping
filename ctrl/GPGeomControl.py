import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from ctrl.GeomControl import GeomControl


class GPGeomControl(GeomControl):
    def __init__(self,
                 model,
                 data,
                 drone_type='cf2',
                 train_x=torch.tensor([0, 0, 0, 0]).float(),
                 train_y=torch.tensor([0]).float()
                 ):
        super().__init__(model, data, drone_type)
        # initialize likelihood and model
        if train_x is not None and train_y is not None:
            self.init_model(train_x, train_y)

    def init_model(self, train_x, train_y):
        # initialize likelihood and model
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(2)]
        self.delta = [ExactGPModel(self.train_x, self.train_y[i], self.likelihood[i]) for i in range(2)]

    def train_model(self):
        training_iter = 200
        self.delta = [ExactGPModel(self.train_x, self.train_y[i], self.likelihood[i]) for i in range(2)]
        # Find optimal model hyperparameters
        for i in range(2):
            self.delta[i].train()
            self.likelihood[i].train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(self.delta[i].parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood[i], self.delta[i])

            for iter in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self.delta[i](self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y[i])
                loss.backward()
                # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                #     iter + 1, training_iter, loss.item(),
                #     self.delta[i].covar_module.base_kernel.lengthscale.item(),
                #     self.delta[i].likelihood.noise.item()
                # ))
                # print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                #     iter + 1, training_iter, loss.item(),
                #     self.delta[i].likelihood.noise.item()
                # ))
                optimizer.step()

    def eval_model(self, test_x):
        for i in range(2):
            # Get into evaluation (predictive posterior) mode
            self.delta[i].eval()
            self.likelihood[i].eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = [self.likelihood[i](self.delta[i](test_x)) for i in range(2)]
        return observed_pred

    def eval_plot(self, test_x):
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            observed_pred = self.eval_model(test_x)
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            # ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            ax.set_title('Training with ' + str(len(self.train_x)) + ' data points')

    def _mu_r(self, pos_e, vel_e):
        # t = self.control_counter*self.control_timestep
        # x = pos_e[0]  # TODO
        # if t > 0:
        #     pred = self.eval_model(torch.tensor([x]).float())
        #     return np.array([(-pred[0].mean/100), 0, 0])
        return np.zeros(3), np.zeros(3)

    def _mu_R(self, cur_quat, cur_ang_vel, rot_e, ang_vel_e):
        q = cur_quat[0:2]  # TODO
        w = cur_ang_vel[0:2]
        # pred = self.eval_model(torch.tensor([np.hstack((q, w))]).float())
        pred = self.eval_model(torch.tensor([q[1]]).float())
        # print(float(-pred[0].mean/200))
        # print(float(-pred[1].mean/200))
        return np.array([float(-pred[0].mean/200), float(-pred[1].mean/200), 0]), np.zeros(3)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4)) #+
                                                         #gpytorch.kernels.PolynomialKernel(2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
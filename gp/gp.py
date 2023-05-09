import torch
import gpytorch


# We will use the simplest form of GP model, exact inference
class GP(gpytorch.models.ExactGP):
    def __init__(self):
        super(GP, self).__init__(
            torch.empty(0), 
            torch.empty(0), 
            gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            
    def predict(self, test_x):
        
        # Get into evaluation (predictive posterior) mode
        self.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self(test_x)
            observed_pred = self.likelihood(output)
        return observed_pred
    
    @property
    def gp_param(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
        
    def fit(self, train_x, train_y, learning_rate=0.1, n_epochs=25):
    
        self.set_train_data(inputs=train_x, targets=train_y, strict=False)
        
        # Find optimal model hyperparameters
        self.train()

        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        previous_gp_param = self.gp_param.clone()

        for i in range(n_epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()

            optimizer.step()
            
            if torch.allclose(self.gp_param, previous_gp_param, rtol=1e-05, atol=1e-04):
                print(f"Early stop at iter {i}")
                break
            previous_gp_param = self.gp_param.clone()

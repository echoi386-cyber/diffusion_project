# %%
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
import torch
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
import ot

# Constants for the duration of our use of Gaussian conditional probability paths, to avoid polluting the namespace...
PARAMS = {
    "scale": 15.0,
    "target_scale": 10.0,
    "target_std": 1.0,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns:
            - Dimensionality of the distribution
        """
        pass
        
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, dim)
        """
        pass
class Density(ABC):
    """
    Distribution with tractable density
    """
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log density at x.
        Args:
            - x: shape (batch_size, dim)
        Returns:
            - log_density: shape (batch_size, 1)
        """
        pass
class Gaussian(torch.nn.Module, Sampleable, Density):
    """
    Multivariate Gaussian distribution
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))
        
    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)
class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """
    Two-dimensional Gaussian mixture model, and is a Density and a Sampleable. Wrapper around torch.distributions.MixtureSameFamily.
    """
    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """
        means: shape (nmodes, 2)
        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)
        self.P = torch.randn((20,self.dim)).to(device)
        self.P = self.P/torch.norm(self.P, dim=0, keepdim=True)
        #self.discrete = self.sample_projected(2000).to(device)
        self.discrete = self.sample(2000).to(device)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
                mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
                component_distribution=D.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    def sample_projected(self, num_samples: int) -> torch.Tensor:
        sample = self.distribution.sample(torch.Size((num_samples,))).to(device) 
        return (self.P[None,:,:]@sample[:,:,None]).squeeze()
    
    def sample_discrete(self, num_samples: int) -> torch.Tensor:
        idx = torch.randint(0, self.discrete.shape[0], (num_samples,)).to(device)
        sample = self.discrete[idx].to(device)
        return sample

    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0, seed = 0.0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor([1.0, 0.0])
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale + torch.Tensor([1.0, 0.0]) * x_offset
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)

    @classmethod
    def symmetric_4D(
        cls, nmodes: int, std: float
    ) -> "GaussianMixture":
        # 4D unit hypersphere samples
        means = torch.randn(nmodes,20)              # standard normal in R^2
        means = means / means.norm(dim=1, keepdim=True)  # normalize to unit length

        # isotropic covariance
        covs = torch.diag_embed(torch.ones(nmodes, 20) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (batch_size, 1)
        Returns:
            - drift_coefficient: shape (batch_size, dim)
        """
        pass
class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (batch_size, dim)
            - t: time, shape (batch_size, 1)
        Returns:
            - drift_coefficient: shape (batch_size, dim)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (batch_size, dim)
            - t: time, shape (batch_size, 1)
        Returns:
            - diffusion_coefficient: shape (batch_size, dim)
        """
        pass
class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs,1)
            - dt: time, shape (bs,1)
        Returns:
            - nxt: state at time t + dt (bs, dim)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (batch_size, dim)
            - ts: timesteps, shape (bs, num_timesteps,1)
        Returns:
            - x_final: final state at time ts[-1], shape (batch_size, dim)
        """
        for t_idx in range(len(ts) - 1):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (bs, dim)
            - ts: timesteps, shape (bs, num_timesteps, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, num
            _timesteps, dim)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.ode.drift_coefficient(xt,t) * h
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.sde.drift_coefficient(xt,t) * h + self.sde.diffusion_coefficient(xt,t) * torch.sqrt(h) * torch.randn_like(xt)

class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x), (num_samples, dim)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z = self.sample_conditioning_variable(num_samples) # (num_samples, dim)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (num_samples, dim)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z
        Args:
            - num_samples: the number of samples
        Returns:
            - z: samples from p(z), (num_samples, dim)
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        pass
        
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """ 
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_score: conditional score (num_samples, dim)
        """ 
        pass
def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )
class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.zeros(1,1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1)), torch.ones(1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """ 
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        t = t.unsqueeze(1) # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t) # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)  
class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.ones(1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1)), torch.zeros(1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        pass 

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt beta_t (num_samples, 1)
        """ 
        t = t.unsqueeze(1) # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t) # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)
class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """ 
        return t 
        
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return torch.ones_like(t)
class SquareRootBeta(Beta):
    """
    Implements beta_t = rt(1-t)
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        return torch.sqrt(1-t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return - 0.5 / (torch.sqrt(1 - t) + 1e-4)
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, alpha: Alpha, beta: Beta, data_std: torch.Tensor = None):
        aux_dim = p_data.dim
        p_simple = Gaussian.isotropic(aux_dim, 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
        self.p_data = p_data

        if data_std is None:
            self.data_std = torch.ones(aux_dim).to(device)
        else:
            self.register_buffer("data_std", data_std)

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z ~ p_data(x)
        Args:
            - num_samples: the number of samples
        Returns:
            - z: samples from p(z), (num_samples, dim)
        """
        #return self.p_data.sample_projected(num_samples)
        #return self.p_data.sample_discrete(num_samples)
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        eps = torch.randn_like(z)

        colored_noise = eps * self.data_std
        return self.alpha(t)*z+self.beta(t)*colored_noise 
        
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """ 
        return (self.alpha.dt(t)-self.beta.dt(t)/self.beta(t)*self.alpha(t))*z + self.beta.dt(t)/self.beta(t)*x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_score: conditional score (num_samples, dim)
        """ 
        noise_variance = (self.beta(t)**2) * (self.data_std**2)
        mean=self.alpha(t) * z
        return -(x - mean)/ (noise_variance + 1e-8)

class ConditionalVectorFieldODE(ODE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.path = path
        self.z = z

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t)
class ConditionalVectorFieldSDE(SDE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor, sigma: float):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, ...)
        """
        super().__init__()
        self.path = path
        self.z = z
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t) + 0.5 * self.sigma**2 * self.path.conditional_score(x,z,t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.randn_like(x)
def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        return torch.nn.Sequential(*mlp)
class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)                
class MLPScore(torch.nn.Module):
    """
    MLP-parameterization of the learned score field
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - s_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x,t], dim=-1)
        #M = torch.randn((self.dim,self.dim)).to(device)
        #A = (M - M.T) / 2.0
        return self.net(xt) #+ (A[None,:,:]@x[:,:,None]).squeeze()        
    
    #def forward_curl(self, x: torch.Tensor, t: torch.Tensor):
    #    return self.forward(x,t) + 0.1*self.A@x
   
class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        # Finish
        self.model.eval()            
class ConditionalScoreMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPScore, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        #z = self.path.p_data.sample_discrete(batch_size)
        #z = self.path.p_data.sample_projected(batch_size)
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size,1).to(device)
        x = self.path.sample_conditional_path(z,t)

        target_score = self.path.conditional_score(x, z, t)

        pred_score = self.model(x, t)

        diff = pred_score-target_score
        
        weight = self.path.data_std.detach()
        weighted_diff = diff * weight

        loss = torch.mean(weighted_diff ** 2)
        return loss
class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPVectorField, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        #z = self.path.p_data.sample_discrete(batch_size)
        #z = self.path.p_data.sample_projected(batch_size)
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size,1).to(device)
        x = self.path.sample_conditional_path(z,t)
        loss = torch.nn.MSELoss()
        return loss(self.model(x,t), self.path.conditional_vector_field(x,z,t))

class LearnedVectorFieldODE(ODE):

    def __init__(self, net: MLPVectorField):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (bs, dim)
            - t: (bs, dim)
        Returns:
            - u_t: (bs, dim)
        """
        return self.net(x, t)
class LangevinFlowSDE(SDE):
    def __init__(self, flow_model: MLPVectorField, score_model: MLPScore, sigma: float):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.flow_model(x,t) + 0.5 * self.sigma ** 2 * self.score_model(x, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.randn_like(x)
class ScoreFromVectorField(torch.nn.Module):
    """
    Parameterization of score via learned vector field (for the special case of a Gaussian conditional probability path)
    """
    def __init__(self, vector_field: MLPVectorField, alpha: Alpha, beta: Beta):
        super().__init__()
        self.vector_field = vector_field
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - score: (bs, dim)
        """
        return (self.alpha(t)*self.vector_field(x,t)-self.alpha.dt(t)*x)/(self.beta(t)**2*self.alpha.dt(t)-self.alpha(t)*self.beta.dt(t)*self.beta(t))
class VectorFieldFromScore(torch.nn.Module):
    """
    Parameterization of score via learned vector field (for the special case of a Gaussian conditional probability path)
    """
    def __init__(self, score: MLPScore, alpha: Alpha, beta: Beta):
        super().__init__()
        self.score = score
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - vector_field: (bs, dim)
        """
        return self.alpha.dt(t)/self.alpha(t)*x + (self.beta(t)**2*self.alpha.dt(t)/self.alpha(t)-self.beta.dt(t)*self.beta(t))*self.score(x,t)

def hist2d_samples(samples, ax: Optional[Axes] = None, bins: int = 200, scale: float = 5.0, percentile: int = 99, **kwargs):
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=[[-scale, scale], [-scale, scale]])
    
    # Determine color normalization based on the 99th percentile
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    
    # Plot using imshow for more control
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)
def hist2d_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, bins=200, scale: float = 5.0, percentile: int = 99, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples).detach().cpu() # (ns, 2)
    hist2d_samples(samples, ax, bins, scale, percentile, **kwargs)
def scatter_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    ax.scatter(samples[:,0].cpu(), samples[:,1].cpu(), **kwargs)
def kdeplot_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    sns.kdeplot(x=samples[:,0].cpu(), y=samples[:,1].cpu(), ax=ax, **kwargs)
def imshow_density(density: Density, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], bins: int, ax: Optional[Axes] = None, x_offset: float = 0.0, **kwargs):
    if ax is None:
        ax = plt.gca()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins).to(device) + x_offset
    y = torch.linspace(y_min, y_max, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.imshow(density.cpu(), extent=[x_min, x_max, y_min, y_max], origin='lower', **kwargs)
def contour_density(density: Density, bins: int, scale: float, ax: Optional[Axes] = None, x_offset:float = 0.0, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale + x_offset, scale + x_offset, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.contour(density.cpu(), origin='lower', **kwargs)

def compare_scores(path,flow_model,score_model):
    #######################
    # Change these values #
    #######################
    num_bins = 30
    num_marginals = 4

    #########################
    # Define score networks #
    #########################
    learned_score_model = score_model
    flow_score_model = ScoreFromVectorField(flow_model, path.alpha, path.beta)


    ###############################
    # Plot score fields over time #
    ###############################
    fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 12))
    axes = axes.reshape((2, num_marginals))

    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]

    ts = torch.linspace(0.01, 0.9999, num_marginals).to(device)
    xs = torch.linspace(-scale, scale, num_bins).to(device)
    ys = torch.linspace(-scale, scale, num_bins).to(device)
    xx, yy = torch.meshgrid(xs, ys)
    xx = xx.reshape(-1,1)
    yy = yy.reshape(-1,1)
    xy = torch.cat([xx,yy], dim=-1)

    axes[0,0].set_ylabel("Learned with Score Matching", fontsize=12)
    axes[1,0].set_ylabel("Computed from $u_t^{{\\theta}}(x)$", fontsize=12)
    for idx in range(num_marginals):
        t = ts[idx]
        bs = num_bins ** 2
        tt = t.view(1,1).expand(bs, 1)
        
        # Learned scores
        learned_scores = learned_score_model(xy, tt)
        learned_scores_x = learned_scores[:,0]
        learned_scores_y = learned_scores[:,1]

        ax = axes[0, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), learned_scores_x.detach().cpu(), learned_scores_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$s_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
        

        # Flow score model
        ax = axes
        flow_scores = flow_score_model(xy,tt)
        flow_scores_x = flow_scores[:,0]
        flow_scores_y = flow_scores[:,1]

        ax = axes[1, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), flow_scores_x.detach().cpu(), flow_scores_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$\\tilde{{s}}_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("compare_scores.pdf", bbox_inches="tight")
def compare_vector_fields(path,flow_model,score_model):
    #######################
    # Change these values #
    #######################
    num_bins = 30
    num_marginals = 4

    #########################
    # Define score networks #
    #########################
    learned_flow_model = flow_model 
    score_flow_model = VectorFieldFromScore(score_model, path.alpha, path.beta)

    ###############################
    # Plot score fields over time #
    ###############################
    fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 12))
    axes = axes.reshape((2, num_marginals))

    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]

    ts = torch.linspace(0.01, 0.9999, num_marginals).to(device)
    xs = torch.linspace(-scale, scale, num_bins).to(device)
    ys = torch.linspace(-scale, scale, num_bins).to(device)
    xx, yy = torch.meshgrid(xs, ys)
    xx = xx.reshape(-1,1)
    yy = yy.reshape(-1,1)
    xy = torch.cat([xx,yy], dim=-1)

    axes[0,0].set_ylabel("Learned with Flow Matching", fontsize=12)
    axes[1,0].set_ylabel("Computed from $s_t^{{\\theta}}(x)$", fontsize=12)
    for idx in range(num_marginals):
        t = ts[idx]
        bs = num_bins ** 2
        tt = t.view(1,1).expand(bs, 1)
        
        # Learned scores
        learned_vector_field = learned_flow_model(xy, tt)
        learned_vector_field_x = learned_vector_field[:,0]
        learned_vector_field_y = learned_vector_field[:,1]

        ax = axes[0, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), learned_vector_field_x.detach().cpu(), learned_vector_field_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$u_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
        

        # Flow score model
        ax = axes
        score_flows = score_flow_model(xy,tt)
        score_flows_x = score_flows[:,0]
        score_flows_y = score_flows[:,1]

        ax = axes[1, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), score_flows_x.detach().cpu(), score_flows_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$\\tilde{{u}}_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("compare_vector_fields.pdf", bbox_inches="tight")

def plot_flow(path,flow_model,num_timesteps,output_file):
    #######################
    # Change these values #
    #######################
    num_samples = 1000
    num_marginals = 3

    ##############
    # Setup Plot #
    ##############
    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]
    legend_size=24
    markerscale=1.8

    # Setup figure
    fig, axes = plt.subplots(1,3, figsize=(36, 12))

    ###########################################
    # Graph Samples from Learned Marginal ODE #
    ###########################################
    ax = axes[1]

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Samples from Learned Marginal ODE", fontsize=20)

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    # Construct integrator and plot trajectories
    ode = LearnedVectorFieldODE(flow_model)
    simulator = EulerSimulator(ode)
    x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
    ts = torch.linspace(0.01, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
    xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)

    # Extract every n-th integration step to plot
    every_n = record_every(num_timesteps=num_timesteps, record_every=num_timesteps // num_marginals)
    xts_every_n = xts[:,every_n,:] # (bs, nts // n, dim)
    ts_every_n = ts[0,every_n] # (nts // n,)
    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].item()
        ax.scatter(xts_every_n[:,plot_idx,0].detach().cpu(), xts_every_n[:,plot_idx,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt:.2f}')

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)

    ##############################################
    # Graph Trajectories of Learned Marginal ODE #
    ##############################################
    ax = axes[2]
    ax.set_title("Trajectories of Learned Marginal ODE", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    for traj_idx in range(num_samples // 10):
        ax.plot(xts[traj_idx,:,0].detach().cpu(), xts[traj_idx,:,1].detach().cpu(), alpha=0.5, color='black')

    ################################################
    # Graph Ground-Truth Marginal Probability Path #
    ################################################
    ax = axes[0]
    ax.set_title("Ground-Truth Marginal Probability Path", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].unsqueeze(0).expand(num_samples, 1)
        marginal_samples = path.sample_marginal_path(tt)
        ax.scatter(marginal_samples[:,0].detach().cpu(), marginal_samples[:,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt[0,0].item():.2f}')

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)
        
    plt.savefig(output_file, bbox_inches="tight")
def plot_score(path,flow_model,score_model,num_timesteps,output_file):
    #######################
    # Change these values #
    #######################
    num_samples = 1000
    num_marginals = 3
    sigma = 2.0 # Don't set sigma too large or you'll get numerical issues!

    ##############
    # Setup Plot #
    ##############
    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]
    legend_size = 24
    markerscale = 1.8

    # Setup figure
    fig, axes = plt.subplots(1,3, figsize=(36, 12))

    ###########################################
    # Graph Samples from Learned Marginal SDE #
    ###########################################
    ax = axes[1]
    ax.set_title("Samples from Learned Marginal SDE", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))


    # Construct integrator and plot trajectories
    sde = LangevinFlowSDE(flow_model, score_model, sigma)
    simulator = EulerMaruyamaSimulator(sde)
    x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
    ts = torch.linspace(0.01, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
    xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)

    # Extract every n-th integration step to plot
    every_n = record_every(num_timesteps=num_timesteps, record_every=num_timesteps // num_marginals)
    xts_every_n = xts[:,every_n,:] # (bs, nts // n, dim)
    ts_every_n = ts[0,every_n] # (nts // n,)
    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].item()
        ax.scatter(xts_every_n[:,plot_idx,0].detach().cpu(), xts_every_n[:,plot_idx,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt:.2f}')

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)

    ###############################################
    # Graph Trajectories of Learned Marginal SDE  #
    ###############################################
    ax = axes[2]
    ax.set_title("Trajectories of Learned Marginal SDE", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    for traj_idx in range(num_samples // 10):
        ax.plot(xts[traj_idx,:,0].detach().cpu(), xts[traj_idx,:,1].detach().cpu(), alpha=0.5, color='black')

    ################################################
    # Graph Ground-Truth Marginal Probability Path #
    ################################################
    ax = axes[0]
    ax.set_title("Ground-Truth Marginal Probability Path", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].unsqueeze(0).expand(num_samples, 1)
        marginal_samples = path.sample_marginal_path(tt)
        ax.scatter(marginal_samples[:,0].detach().cpu(), marginal_samples[:,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt[0,0].item():.2f}')

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)
        
    plt.savefig(output_file, bbox_inches="tight")
def wasserstein_distance(samples, target_samples):
    n = len(samples)
    # Uniform weights for empirical distributions
    a = torch.ones(n,device=device) / n
    b = torch.ones(n,device=device) / n

    # Pairwise cost matrix (Euclidean distances)
    M = ot.dist(samples, target_samples, metric='euclidean')  # shape (n, n)

    # --- Solve optimal transport ---
    res = ot.solve(M, a, b)
    return res.value
def simulate_flow(path,flow_model,num_samples,num_timesteps):
    # Construct integrator and plot trajectories
    ode = LearnedVectorFieldODE(flow_model)
    simulator = EulerSimulator(ode)
    x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
    ts = torch.linspace(0.01, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
    xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)
    x1 = xts[:,-1,:]
    return x1
def simulate_score(path,flow_model,score_model,num_samples,num_timesteps):
    # Construct integrator and plot trajectories
    sigma = 2.0
    sde = LangevinFlowSDE(flow_model,score_model,sigma)
    simulator = EulerMaruyamaSimulator(sde)
    x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
    ts = torch.linspace(0.01, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
    xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)
    x1 = xts[:,-1,:]
    return x1
def plot_results(results):
    # -------------------------------------------------------
    # Plot Wasserstein vs. timesteps (from `results` dict)
    # -------------------------------------------------------
    plt.figure(figsize=(6,5))

    # plot flow model curve
    if "flow_deterministic" in results:
        steps, Ws = results["flow_deterministic"]
        plt.plot(steps, Ws, marker="o", linewidth=2, label="Flow model (deterministic)")

    if "flow_stochastic" in results:
        steps, Ws = results["flow_stochastic"]
        plt.plot(steps, Ws, marker="o", linewidth=2, label="Flow model (stochastic)")

    # plot score model curve
    if "score_stochastic" in results:
        steps, Ws = results["score_stochastic"]
        plt.plot(steps, Ws, marker="s", linewidth=2, label="Score model (stochastic)")

    if "score_deterministic" in results:
        steps, Ws = results["score_deterministic"]
        plt.plot(steps, Ws, marker="s", linewidth=2, label="Score model (deterministic)")

    plt.ylabel("Error (Wasserstein-1 distance)", fontsize=14)
    plt.xlabel("Iterations (timesteps)", fontsize=14)
    plt.title("Convergence of Simulation Error vs Iterations", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # -------------------------------------------------------
    # Save plot as PDF
    # -------------------------------------------------------
    plt.savefig("./convergence.pdf", bbox_inches="tight")

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from pdb import set_trace

def NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)

def gram_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    N, Z = z1.shape 
    device = z1.device 

    representations = torch.cat([z1, z2], dim=0)
    dsq_all = euclidsq(representations, representations)
    sigma = torch.sqrt(torch.median(dsq_all)).item()
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1) + 1


    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)

    negatives = torch.matmul(negatives, torch.ones(negatives.shape[-1], 1, device=negatives.device))

    # loss = -1.0 * (torch.log(negatives) - torch.log(positives)).sum()
    loss = torch.log(positives / negatives).sum()

    # logits = torch.cat([positives, negatives], dim=1)
    # logits /= temperature
    # labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    # loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)

############################################################################
# GramNet Ratio Loss & Utilies
############################################################################

# Returns euclidean distance squared
def euclidsq(x, y):
    return torch.pow(torch.cdist(x, y), 2)

# Returns gaussian kernel calculation
def gaussian_gramian(esq, σ):
    return torch.exp(torch.div(-esq, 2 * σ**2))
    
# Returns euclidean distances for denom & numerator
def get_esq(x_de, x_nu):
    return euclidsq(x_de, x_de), euclidsq(x_de, x_nu), euclidsq(x_nu, x_nu)

def kmm_ratios(Kdede, Kdenu, eps_ratio=0.0, version='original'):
    if version == 'original':
        n_de, n_nu = Kdenu.shape
        if eps_ratio > 0:
            A = Kdede + eps_ratio * torch.eye(n_de).to(Kdede.device)
        else:
            A = Kdede
        B = Kdenu

        # return torch.matmul(torch.matmul(torch.inverse(A), B), torch.ones(n_nu, 1).to(Kdede.device))
        return (n_de / n_nu) * torch.matmul(B/A[0][0], torch.ones(n_nu, 1).to(Kdede.device))
    elif version == 'efficient':
        _, n_nu = Kdenu.shape
        
        if eps_ratio > 0:
            A = Kdede + eps_ratio * torch.ones(n_de).to(Kdede.device)
        else:
            A = Kdede
        
        B = Kdenu

        # 2 / 2 * (N-1) == 1 / (N - 1), where N is the batch size
        return (1 / n_nu) * (torch.matmul(B, torch.ones(B.shape[-1], device=B.device)) / A)

def mmd_loss(z1, z2, σs=[], eps_ratio=0.0, clip_ratio=False, version='original'):
    # Note that original & efficient versions assume different z1, z2 distributions, so be careful
    if version == 'original':
        return mmd_loss_original(z1, z2, σs, eps_ratio, clip_ratio)
    elif version == 'efficient':
        return mmd_loss_efficient(z1, z2, σs, eps_ratio, clip_ratio)

# Efficient version of gramnet ratio loss that performs all calculations at once
# and then re-arranges for calculating the ratio
def mmd_loss_efficient(z1, z2, σs=[], eps_ratio=0.0, clip_ratio=False):
    # Assuming z1, z2 are transformed views of x (N, dim_z)

    N = z1.shape[0]
    assert N == z2.shape[0]

    all_z = torch.cat([z1, z2], dim=0)
    dsq_all = euclidsq(all_z, all_z)

    # Creating list of sigmas, if not defined
    if len(σs) == 0:
        # A heuristic is to use the median of pairwise distances as σ, suggested by Sugiyama's book
        # TODO: Ask about this sigma
        sigma = torch.sqrt(
            torch.median(dsq_all)
        ).item()

        σs.append(sigma)
        # σs.append(sigma * 0.333)
        # σs.append(sigma * 0.2)
        # σs.append(sigma / 0.2)
        # σs.append(sigma / 0.333)
    
    ratio = 0.0

    for σ in σs:
        K_all = gaussian_gramian(dsq_all, σ)
        
        # Clipping kernel
        K_all = torch.clamp(K_all, min=1e-10, max=1e15)
        # Getting the N'th diagonal above and below, but should be symmetrical/equal
        # Ordered like 11', 22' ... NN', 1'1... N'N
        # Shape (2*N)
        Kdede = torch.cat([torch.diag(K_all, N), torch.diag(K_all, -N)], dim=0)

        # Creating mask for positives
        diag = torch.eye(2*N, dtype=torch.bool, device=K_all.device)
        diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

        # Using opposite of positive mask to get all negatives
        # Ordered like (12, 13, ... 1N'), (21, 23, ... 2N') in each row (same ordering as positives)
        # Shape (2*N, 2*(N-1))
        Kdenu = K_all[~diag].view(2*N, -1)

        ratio += kmm_ratios(Kdede, Kdenu, eps_ratio, version='efficient')
    
    ratio = ratio / len(σs)
    # ratio = torch.relu(ratio) if clip_ratio else ratio # Clip ratio on the upper-bound 
    # ratio = torch.clip(ratio, min=0, max=1e15)
    
    # mmd = torch.sqrt(torch.relu(mmdsq))

    # pearson_div = -1.0 * torch.mean(torch.pow(ratio - 1, 2))
    pearson_div = -1.0 * torch.mean(ratio)
    
    return pearson_div

# Un-optimized version of gramnet loss from Kai's pytorch implementation
def mmd_loss_original(z1, z2, σs=[], eps_ratio=0.0, clip_ratio=False):
    # Assuming z1 = q (2, dim_z), z2 = p (N-1, dim_z)

    dsq_dede, dsq_denu, dsq_nunu = get_esq(z1, z2)

    # Creating list of sigmas, if not defined
    if len(σs) == 0:
        # A heuristic is to use the median of pairwise distances as σ, suggested by Sugiyama's book
        sigma = torch.sqrt(
            torch.median(
                torch.cat([dsq_dede.squeeze(), dsq_denu.squeeze(), dsq_nunu.squeeze()], 1)
            )
        ).item()

        σs.append(sigma)
        σs.append(sigma * 0.333)
        σs.append(sigma * 0.2)
        σs.append(sigma / 0.2)
        σs.append(sigma / 0.333)
    
    ratio = 0.0

    for σ in σs:
        Kdede = gaussian_gramian(dsq_dede, σ)
        Kdenu = gaussian_gramian(dsq_denu, σ)
        # Knunu = gaussian_gramian(dsq_nunu, σ)

        ratio += kmm_ratios(Kdede, Kdenu, eps_ratio)
    
    ratio = ratio / len(σs)
    ratio = torch.relu(ratio) if clip_ratio else ratio

    pearson_div = torch.mean(torch.pow(ratio - 1, 2)) + ratio.sum()
    
    return pearson_div

    

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

############################################################################
# DRE Formulation
############################################################################
class Discriminator(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512):
        super().__init__()
        ''' 
        Discriminator for estimating ratio (joint / marginal)
        '''

        self.in_dim = in_dim
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)

        return x 

class SimCLR(nn.Module):

    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )



    def forward(self, x1, x2, sym_loss_weight=1.0, logistic_loss_weight=0.0):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        loss = NT_XentLoss(z1, z2)
        return {'loss':loss, 'loss_sym': loss}

class SimCLRGram(nn.Module):
    def __init__(self, backbone=resnet50(), proj_dim=128):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim, out_dim=proj_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

    def forward(self, x1, x2, sym_loss_weight=1.0, logistic_loss_weight=0.0):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        loss = gram_loss(z1, z2)
        return {'loss':loss, 'loss_gram': loss}

# Original SimCLR model with a discriminator added for only estimating MI (no gradients)
class SimCLRMI(nn.Module):
    def __init__(self, backbone=resnet50(), proj_dim=128):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim, out_dim=proj_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.discriminator = Discriminator(in_dim=proj_dim*2)

    def forward(self, x1, x2, disc=False):
        if disc:
            return self.forward_d(x1, x2)
        
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        loss = NT_XentLoss(z1, z2)
        return {'loss':loss, 'loss_sym': loss}   
    
    def forward_d(self, x1, x2):
        d = self.discriminator
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        real = torch.ones((x1.shape[0], 1), dtype=torch.float32, device=x1.device)
        fake = torch.zeros((x1.shape[0], 1), dtype=torch.float32, device=x1.device)

        real_outputs = d(torch.cat((z1, z2), dim=-1))
        fake_outputs = d(torch.cat((z1[torch.randperm(z1.size()[0])], z2[torch.randperm(z2.size()[0])]), dim=-1))
        
        real_loss = F.binary_cross_entropy(real_outputs, real)
        fake_loss = F.binary_cross_entropy(fake_outputs, fake)

        d_loss = (real_loss + fake_loss) / 2

        mi = -1.0 * real_outputs
        
        return {'loss_d/total': d_loss, 'loss_d/real': real_loss, 'loss_d/fake': fake_loss, 'loss_d/mi': mi}

class SimCLRJoint(nn.Module):

    def __init__(self, backbone=resnet50(), proj_dim=128):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(in_dim=backbone.output_dim, out_dim=proj_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.discriminator = Discriminator(in_dim=proj_dim*2)

    def forward(self, x1, x2, sym_loss_weight=1.0, logistic_loss_weight=1.0, est=False):
        # MI Estimation
        if est:
            return self.forward_est(x1, x2, logistic_loss_weight)
        # MI Maximization
        else:
            return self.forward_max(x1, x2, sym_loss_weight)

    def forward_est(self, x1, x2, logistic_loss_weight=1.0):
        d = self.discriminator
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        if logistic_loss_weight > 0.0:
            real = torch.ones((x1.shape[0], 1), dtype=torch.float32, device=x1.device)
            fake = torch.zeros((x1.shape[0], 1), dtype=torch.float32, device=x1.device)

            real_outputs = d(torch.cat((z1, z2), dim=-1))
            fake_outputs = d(torch.cat((z1[torch.randperm(z1.size()[0])], z2[torch.randperm(z2.size()[0])]), dim=-1))
            
            real_loss = F.binary_cross_entropy(real_outputs, real)
            fake_loss = F.binary_cross_entropy(fake_outputs, fake)

            d_loss = ((real_loss + fake_loss) / 2 * logistic_loss_weight) if logistic_loss_weight > 0.0 else 0.0
        else:
            raise Exception('Logistic loss weight undefined for forward pass for MI estimation')
        
        return {'loss_d': d_loss, 'loss_d_real': real_loss, 'loss_d_fake': fake_loss}
    
    def forward_max(self, x1, x2, sym_loss_weight=1.0):
        d = self.discriminator
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        sym_loss = (NT_XentLoss(z1, z2) * sym_loss_weight) if sym_loss_weight > 0.0 else 0.0
        mi_loss = -1.0 * d(torch.cat((z1, z2), dim=-1))

        return {'loss_m': sym_loss + mi_loss, 'loss_sym': sym_loss, 'loss_mi': mi_loss}
        



















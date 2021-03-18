import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


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
    
    def forward_max(x1, x2, sym_loss_weight=1.0):
        d = self.discriminator
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        sym_loss = (NT_XentLoss(z1, z2) * sym_loss_weight) if sym_loss_weight > 0.0 else 0.0
        mi_loss = -1.0 * d(torch.cat((z1, z2), dim=-1))

        return {'loss_m': sym_loss + mi_loss, 'loss_sym': sym_loss, 'loss_mi': mi_loss}
        



















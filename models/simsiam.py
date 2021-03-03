import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50

############################################################################
# SimSiam Loss
############################################################################

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

############################################################################
# MMD Loss / Utils
############################################################################
def euclidsq(x, y):
    return torch.pow(torch.cdist(x, y), 2)

def gaussian_gramian(esq, σ):
    return torch.exp(torch.div(-esq, 2 * σ**2))

def prepare(x_de, x_nu):
    return euclidsq(x_de, x_de), euclidsq(x_de, x_nu), euclidsq(x_nu, x_nu)

def kmm_ratios(Kdede, Kdenu, λ, use_solve=False):
    n_de, n_nu = Kdenu.shape
    if λ > 0:
        A = Kdede + λ * torch.eye(n_de).to(Kdenu.device)
    else:
        A = Kdede
    # Equivalent implement based on 1) solver and 2) matrix inversion
    if use_solve:
        B = torch.sum(Kdenu, 1, keepdim=True)
        return (n_de / n_nu) * torch.solve(B, A).solution
    else:
        B = Kdenu
        return torch.matmul(torch.matmul(torch.inverse(A), B), torch.ones(n_nu, 1).to(Kdenu.device))

def mmdsq_of(Kdede, Kdenu, Knunu):
    return torch.mean(Kdede) - 2 * torch.mean(Kdenu) + torch.mean(Knunu)

def estimate_ratio_compute_mmd(x_de, x_nu, σs=[1, 10, 100, 1000], clip_ratio=True, eps_ratio=0.001):
    dsq_dede, dsq_denu, dsq_nunu = prepare(x_de, x_nu)
    if len(σs) == 0:
        # A heuristic is to use the median of pairwise distances as σ, suggested by Sugiyama's book
        sigma = torch.sqrt(
            torch.median(
                torch.cat([dsq_dede.squeeze(), dsq_denu.squeeze(), dsq_nunu.squeeze()], 1)
            )
        ).item()
        # if not opt.nowandb:
        #     wandb.log({"heuristic_sigma" : sigma})
        # elif opt.monitor_heuristic:
        #     print("heuristic sigma: ", sigma)
        # Use [sigma / 5, sigma / 3, sigma, sigma * 3, sigma * 5] if nothing provided
        if len(σs) == 0:
            σs.append(sigma)
            σs.append(sigma * 0.333)
            σs.append(sigma * 0.2)
            σs.append(sigma / 0.2)
            σs.append(sigma / 0.333)
    
    is_first = True
    ratio = None
    mmdsq = None
    for σ in σs:
        Kdede = gaussian_gramian(dsq_dede, σ)
        Kdenu = gaussian_gramian(dsq_denu, σ)
        Knunu = gaussian_gramian(dsq_nunu, σ)
        if is_first:
            ratio = kmm_ratios(Kdede, Kdenu, eps_ratio)
            mmdsq = mmdsq_of(Kdede, Kdenu, Knunu)
            is_first = False
        else:
            ratio += kmm_ratios(Kdede, Kdenu, eps_ratio)
            mmdsq += mmdsq_of(Kdede, Kdenu, Knunu)
    
    ratio = ratio / len(σs)
    ratio = torch.relu(ratio) + eps_ratio if clip_ratio else ratio
    mmd = torch.sqrt(torch.relu(mmdsq))
    
    return ratio, mmd


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}

############################################################################
# Knowledge Distillation Models
############################################################################

class SimSiamKD(nn.Module):
    def __init__(self, backbones):
        super().__init__()
        
        self.backbone_s, self.backbone_t = backbones
        self.projector = projection_MLP(self.backbone_s.output_dim)

        # Student encoder
        self.encoder_s = nn.Sequential( # f encoder
            self.backbone_s,
            self.projector
        )

        # Teacher encoder
        self.encoder_t =  nn.Sequential( # f encoder
            self.backbone_t,
            # self.projector
        )
        self.predictor = prediction_MLP(in_dim=self.projector.out_dim, out_dim=self.backbone_t.output_dim)
    
    def forward(self, x1, x2):

        f_s, f_t, h = self.encoder_s, self.encoder_t, self.predictor
        z1_1, z1_2, z2_1, z2_2 = f_s(x1), f_s(x2), f_t(x1), f_t(x2)
        p1, p2 = h(z1_1), h(z1_2)
        L = D(p1, z2_1) / 2 + D(p2, z2_2) / 2
        return {'loss': L}

class SimSiamKDAnchor(nn.Module):
    def __init__(self, backbones=[resnet50, resnet50]):
        super().__init__()
        
        self.backbone_s, self.backbone_t = backbones
        self.projector = projection_MLP(in_dim=self.backbone_s.output_dim)

        # Student encoder
        self.encoder_s = nn.Sequential( # f encoder
            self.backbone_s,
            self.projector
        )

        # Teacher encoder
        self.encoder_t =  nn.Sequential( # f encoder
            self.backbone_t,
            # self.projector
        )
        self.predictor = prediction_MLP(in_dim=self.projector.out_dim, out_dim=self.backbone_t.output_dim)
    
    def forward(self, x1, x2, x3):

        f_s, f_t, h = self.encoder_s, self.encoder_t, self.predictor
        z1_1, z1_2, z2 = f_s(x1), f_s(x2), f_t(x3)
        p1, p2 = h(z1_1), h(z1_2)
        L = D(p1, z2) / 2 + D(p2, z2) / 2
        return {'loss': L}

############################################################################
# Adversarial Formulation
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

class SimSiamAdv(nn.Module):
    def __init__(self, backbone=resnet50(), proj_dim=128):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(in_dim=backbone.output_dim, out_dim=proj_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )

        self.discriminator = Discriminator(in_dim=proj_dim*2)

    def forward(self, x1, x2, disc=False):
        if not disc:
            return self.forward_e(x1, x2)
        else:
            return self.forward_d(x1, x2)
        
    def forward_e(self, x1, x2):

        f, d = self.encoder, self.discriminator
        z1, z2 = f(x1), f(x2)
        L = -1.0 * d(torch.cat((z1, z2), dim=-1))
        
        return {'loss_e': L}
    
    def forward_d(self, x1, x2):
        f, d = self.encoder, self.discriminator
        z1, z2 = f(x1), f(x2)
        
        real = torch.ones((x1.shape[0], 1), dtype=torch.float32, device=x1.device)
        fake = torch.zeros((x1.shape[0], 1), dtype=torch.float32, device=x1.device)

        real_outputs = d(torch.cat((z1, z2), dim=-1))
        fake_outputs = d(torch.cat((z1[torch.randperm(z1.size()[0])], z2[torch.randperm(z2.size()[0])]), dim=-1))
        
        real_loss = F.binary_cross_entropy(real_outputs, real)
        fake_loss = F.binary_cross_entropy(fake_outputs, fake)

        return {'loss_d': (real_loss + fake_loss) / 2, 'loss_d_real': real_loss, 'loss_d_fake': fake_loss}

class SimSiamAdvMMD(nn.Module):
    def __init__(self, backbone=resnet50(), proj_dim=128):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(in_dim=backbone.output_dim, out_dim=proj_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )

    def forward(self, x1, x2, disc=False):
        if not disc:
            return self.forward_e(x1, x2)
        else:
            return self.forward_d(x1, x2)
        
    def forward_e(self, x1, x2):

        f = self.encoder
        z1, z2 = f(x1), f(x2)
        L = self.forward_d(z1, z2)
        
        return {'loss_e': L}
    
    def forward_d(self, z1, z2):
        f, d = self.encoder, self.discriminator
        z1, z2 = f(x1), f(x2)
        z1_shuffled, z2_shuffled = z1[torch.randperm(z1.size()[0])], z2[torch.randperm(z2.size()[0])]
        num_ratio, num_mmd = estimate_ratio_compute_mmd(torch.cat((z1, z2), dim=-1), torch.cat((z1_shuffled, z2_shuffled), dim=-1))
        num_ratio = 1.0 / num_ratio

        loss = 0.0
        for i in range(len(num_ratio)):
            denum_ratio, denum_mmd = estimate_ratio_compute_mmd(torch.cat((z1[i].repeat(z1.shape[0], 1), z2_shuffled), dim=-1), torch.cat((z2, z2), dim=-1))
            loss += (-1.0 * torch.log(num_ratio[i] / torch.sum(denum_ratio)))

        return loss
    

if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469













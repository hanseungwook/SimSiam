import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    elif version == 'symmetric':
        return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception

def l2_metric(p, z):
    return F.mse_loss(p, z)


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
            nn.BatchNorm1d(hidden_dim)
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

class Embed(nn.Module):
    """Embedding module from Contrastive Representation Distillation 
    https://arxiv.org/pdf/1910.10699.pdf (Page 4)

    Linearly transforms predicted embeddings into the same dimension between student and teacher
    """
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        # self.l2norm = Normalize(2)

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        x = self.linear(x)
        # x = self.l2norm(x)
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

class SimSiamNoSG(nn.Module):
    def __init__(self, backbone=[resnet50(), resnet50(), resnet50()]):
        super().__init__()
        
        self.backbone1, self.backbone2, self.backbone3 = backbone
        self.projector1 = projection_MLP(self.backbone1.output_dim)
        self.projector2 = projection_MLP(self.backbone2.output_dim)
        self.projector3 = projection_MLP(self.backbone3.output_dim)

        self.encoder1 = nn.Sequential( # f encoder
            self.backbone1,
            self.projector1
        )

        self.encoder2 = nn.Sequential( # f encoder
            self.backbone2,
            self.projector2
        )

        self.encoder3 = nn.Sequential( # f encoder
            self.backbone3,
            self.projector3
        )

        self.predictor1 = prediction_MLP()
        self.predictor2 = prediction_MLP()
        self.predictor3 = prediction_MLP()
    
    def forward(self, x1, x2, x3):
        # Select uniform sampling from 3 symmetric pairs
        pair_idxs = torch.randint(0, 3, (x1.shape[0],))

        total_L = []
        # Iterating through all possible symmetric pairs
        for pair_idx in range(3):
            f, f_h, g, g_h, v1, v2 = self.get_pair_encoders_views(pair_idx, x1, x2, x3)

            # Selecting respective pairs of views/images from mini-batch that were assigned to this symmetric pair optimization
            v1 = v1[torch.where(pair_idxs == pair_idx)[0]]
            v2 = v2[torch.where(pair_idxs == pair_idx)[0]]
            z1, z2 = f(v2), g(v2)
            p1, p2 = f_h(z1), g_h(z2)
            L = D(p1, z2) / 2 + D(p2, z1) / 2
            L.backward()
            total_L.append(L)

        return {'loss': sum(total_L) / len(total_L)}
    
    def get_pair_encoders_views(self, pair_idx, x1, x2, x3):
        e1, e1_p, e2, e2_p = None, None, None, None
        v1, v2 = None, None

        if pair_idx == 0:
            e1, e1_p, e2, e2_p = self.encoder1, self.predictor1, self.encoder2, self.predictor2
            v1, v2 = x1, x2
        elif pair_idx == 1:
            e1, e1_p, e2, e2_p = self.encoder1, self.predictor1, self.encoder3, self.predictor3
            v1, v2 = x1, x3
        elif pair_idx == 2:
            e1, e1_p, e2, e2_p = self.encoder2, self.predictor2, self.encoder3, self.predictor3
            v1, v2 = x2, x3
        else: 
            raise NotImplementedError('Respective pair idx not implemented: {}'.format(pair_idx))

        return e1, e1_p, e2, e2_p, v1, v2    
        

class SimSiamKD(nn.Module):
    def __init__(self, backbones):
        super().__init__()
        
        self.backbone_s, self.backbone_t = backbones
        self.projector = projection_MLP(self.backbone_t.output_dim)
        self.embed = Embed(dim_in=self.backbone_s.output_dim, dim_out=self.backbone_t.output_dim)

        # Student encoder
        self.encoder_s = nn.Sequential( # f encoder
            self.backbone_s,
            self.embed,
            self.projector
        )

        # Teacher encoder
        self.encoder_t =  nn.Sequential( # f encoder
            self.backbone_t,
            self.projector
        )

        self.predictor = prediction_MLP(in_dim=self.projector.out_dim)
    
    def forward(self, x1, x2):

        f_s, f_t, h = self.encoder_s, self.encoder_t, self.predictor
        z1_1, z1_2, z2_1, z2_2 = f_s(x1), f_s(x2), f_t(x1), f_t(x2)
        p1, p2 = h(z1_1), h(z1_2)
        L = D(p1, z2_1) / 2 + D(p2, z2_2) / 2
        L2 = (l2_metric(p1, z2_1) + l2_metric(p2, z2_2)) / 2 

        return {'loss': L, 'l2': L2}

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













import torch
import torch.nn.functional as F

def reparameterize(mean,lnvar):
	sig = torch.exp(lnvar/2.)
	eps = torch.randn_like(sig)
	return eps.mul_(sig).add_(mean)

def multinm_log_likelihood(xx,pr, eps=1e-8):
	return torch.sum(xx * torch.log(pr+eps),dim=-1)

def multi_dir_log_likelihood(x,alpha):
	a = torch.lgamma(alpha.sum(1)) - torch.lgamma(alpha).sum(1)
	b = torch.lgamma(x + alpha).sum(1) - torch.lgamma( (x + alpha).sum(1))
	return a + b 


def kl_loss(mean,lnvar):
	return  -0.5 * torch.sum(1. + lnvar - torch.pow(mean,2) - torch.exp(lnvar), dim=-1)


def cl_loss(anchor, positive, negative, margin=1.0):
    # distance_positive = (anchor - positive).pow(2).sum(1)  
    # distance_negative = (anchor - negative).pow(2).sum(1)  

    distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
    distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)

    losses = F.relu(distance_positive - distance_negative + margin)

    return losses.mean()


def pcl_loss(z_i, z_j,temperature = 1.0):
    
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / temperature)
        denominator = mask * torch.exp(similarity / temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scvi.distributions import ZeroInflatedNegativeBinomial


def get_zinb_reconstruction_loss(x, px_s, px_r, px_d):
    '''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
    return torch.mean(-ZeroInflatedNegativeBinomial(mu=px_s, theta=px_r, zi_logits=px_d).log_prob(x).sum(dim=-1))

def minimal_overlap_loss(z_common, z_unique):
    z_common_norm = F.normalize(z_common, p=2, dim=-1)
    z_unique_norm = F.normalize(z_unique, p=2, dim=-1)
    cosine_similarity = torch.sum(z_common_norm * z_unique_norm, dim=-1)
    # return torch.mean(torch.abs(cosine_similarity))
    return torch.mean(cosine_similarity)


def entropy_loss(h1, h2):

    h1_normalized = F.normalize(h1, p=2, dim=1)
    h2_normalized = F.normalize(h2, p=2, dim=1)

    similarity = F.cosine_similarity(h1_normalized.unsqueeze(1), h2_normalized.unsqueeze(0), dim=2)

    entropy = -torch.mean(torch.log(similarity + 1e-6))  

    return entropy

from sklearn.neighbors import NearestNeighbors

def compute_weights(z, k=15):
    z_numpy = z.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(z_numpy)
    distances, _ = nbrs.kneighbors(z_numpy)
    distances = torch.tensor(distances, device=z.device) 
    density = torch.exp(-distances).sum(dim=1)  
    inverse_density = 1.0 / (density + 1e-6)
    weights = inverse_density / torch.mean(inverse_density)
    return weights

def pcl_loss_with_weighted_nbr(z_i, z_j, temperature=1.0, k=15):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim_ij = torch.diag(similarity, batch_size)
    sim_ji = torch.diag(similarity, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)
    mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
    numerator = torch.exp(positives / temperature)
    denominator = mask * torch.exp(similarity / temperature)
    weights = compute_weights(torch.cat([z_i, z_j], dim=0), k)
    numerator = numerator * weights
    denominator = denominator * weights.unsqueeze(0)
    all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
    loss = torch.sum(all_losses) / (2 * batch_size)
    return loss


def identify_rare_groups(latent_space, num_clusters=5, rare_group_threshold=0.1):
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(latent_space.cpu().detach().numpy())
    
    cluster_counts = torch.tensor([(cluster_labels == i).sum() for i in range(num_clusters)])
    total_samples = len(cluster_labels)
    rare_clusters = cluster_counts < (rare_group_threshold * total_samples)
    
    return torch.tensor(cluster_labels, device=latent_space.device), rare_clusters

def pcl_loss_with_rare_cluster(z_i, z_j, num_clusters=10, rare_group_threshold=0.05, rare_group_weight=2.5,temperature=1.0):
    batch_size = z_i.size(0)

    z = torch.cat([z_i, z_j], dim=0)
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity, batch_size)
    sim_ji = torch.diag(similarity, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
    numerator = torch.exp(positives / temperature)
    denominator = mask * torch.exp(similarity / temperature)

    all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))

    cluster_labels, rare_clusters = identify_rare_groups(z, num_clusters, rare_group_threshold)
    
    weights = torch.ones_like(all_losses, device=z_i.device)
    for cluster_idx in range(num_clusters):
        if rare_clusters[cluster_idx]:
            cluster_indices = (cluster_labels == cluster_idx).nonzero(as_tuple=True)[0]
            weights[cluster_indices] = rare_group_weight
    weighted_losses = all_losses * weights
    loss = torch.sum(weighted_losses) / (2 * batch_size)

    return loss

def pcl_loss_with_weighted_cluster(z_i, z_j, num_clusters=10, unmatched_group_weight=2.0,temperature=1.0):

    batch_size = z_i.size(0)
    
    z = torch.cat([z_i, z_j], dim=0)
    
    z_np = z.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(z_np)
    cluster_assignments = kmeans.labels_

    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity, batch_size)
    sim_ji = torch.diag(similarity, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()

    weight_matrix = torch.ones_like(similarity, device=z.device)

    for i in range(batch_size * 2):
        for j in range(batch_size * 2):
            if i != j:  
                if cluster_assignments[i] != cluster_assignments[j]:
                    weight_matrix[i, j] += unmatched_group_weight
      

    numerator = torch.exp(positives / temperature)
    denominator = mask * torch.exp(similarity / temperature) * weight_matrix

    all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
    loss = torch.sum(all_losses) / (2 * batch_size)

    return loss

def pcl_loss_with_margin(z_i, z_j, margin=0.5, temperature=1.0):

    batch_size = z_i.size(0)

    z = torch.cat([z_i, z_j], dim=0)

    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    
    sim_ij = torch.diag(similarity, batch_size)
    sim_ji = torch.diag(similarity, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    distance = 1 - similarity  

    mask = (~torch.eye(batch_size * 2, dtype=torch.bool, device=z.device)).float()
    margin_loss = F.relu(distance - margin) * mask  
    
    numerator = torch.exp(positives / temperature)
    
    denominator = torch.sum(torch.exp(-margin_loss / temperature), dim=1)
    
    all_losses = -torch.log(numerator / denominator)
    loss = torch.sum(all_losses) / (2 * batch_size)

    return loss

def pcl_loss_base(z_i, z_j,temperature = 1.0):
    
        batch_size = z_i.size(0)

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
  
def pcl_loss_with_triplet(z_i, z_j, margin=1.0, temperature=1.0):
    batch_size = z_i.size(0)

    z = torch.cat([z_i, z_j], dim=0)
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity, batch_size)
    sim_ji = torch.diag(similarity, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
    numerator = torch.exp(positives / temperature)
    denominator = mask * torch.exp(similarity / temperature)

    all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
    contrastive_loss = torch.sum(all_losses) / (2 * batch_size)


    anchor = z_i  
    positive = z_j  

    negatives_idx = torch.cat([
        torch.arange(0, batch_size), 
        torch.arange(batch_size, 2 * batch_size)  
    ])
    negatives_idx = negatives_idx[torch.randperm(negatives_idx.size(0))]  

    negative = z[negatives_idx[:batch_size]]

    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)

    triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()


    total_loss = contrastive_loss + triplet_loss
    return total_loss

def pcl_loss(z_i, z_j,mode):
    
    loss = None 
    
    if mode == 'weighted':
        loss = pcl_loss_with_weighted_cluster(z_i,z_j)
    elif mode == 'rare':
        loss = pcl_loss_with_rare_cluster(z_i,z_j)
    elif mode == 'margin':
        loss = pcl_loss_with_margin(z_i,z_j)
    elif mode == 'wnbr':
        loss = pcl_loss_with_weighted_nbr(z_i,z_j)
    else:
        loss = pcl_loss_base(z_i,z_j)
    
    return loss


def attention_entropy(attention_weights):
    l = -torch.mean(torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1))
    return l

def cosine_similarity_loss(h1,h2):
    batch_size = h1.size(0)
    all_loss =1- F.cosine_similarity(h1, h2)
    loss = torch.sum(all_loss) / (2 * batch_size)
    return loss
    
def cosine_similarity_loss_with_margin(h1, h2, margin=0.2):
    cos_sim = F.cosine_similarity(h1, h2)
    return torch.clamp(1 - cos_sim - margin, min=0).mean()

def mse_similarity_loss(h1, h2):
    return F.mse_loss(h1, h2)



def mse_similarity_loss_with_shuffle(h1, h2):

    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    
    shuffled_indices_h1 = torch.randperm(h1.size(0))
    shuffled_indices_h2 = torch.randperm(h2.size(0))
    
    h1_shuffled = h1[shuffled_indices_h1]
    h2_shuffled = h2[shuffled_indices_h2]
    
    sim1 = torch.sum(h1 * h1_shuffled, dim=1) 
    sim2 = torch.sum(h2 * h2_shuffled, dim=1) 

    loss = F.mse_loss(sim1, sim2)
    
    return loss


def correlation_loss(h1, h2):
    
    batch_size = h1.size(0)

    h1_centered = h1 - h1.mean(dim=0, keepdim=True)
    h2_centered = h2 - h2.mean(dim=0, keepdim=True)

    covariance = (h1_centered * h2_centered).sum(dim=0)

    h1_std = torch.sqrt((h1_centered ** 2).sum(dim=0) + 1e-8)
    h2_std = torch.sqrt((h2_centered ** 2).sum(dim=0) + 1e-8)

    correlation = covariance / (h1_std * h2_std)

    all_losses = correlation.sum()
    
    loss = torch.sum(all_losses) / (2 * batch_size)

    return loss


def symmetric_kl_loss(h1, h2):
    h1 = F.softmax(h1, dim=1)  
    h2 = F.softmax(h2, dim=1)  
    kl_forward = F.kl_div(h1.log(), h2, reduction="batchmean")
    kl_backward = F.kl_div(h2.log(), h1, reduction="batchmean")
    return 0.5 * (kl_forward + kl_backward)


def reconstuction_loss(h1, h2):

    cov_h1 = h1.T @ h1
    cov_h2 = h2.T @ h2

    u1, _, _ = torch.svd(cov_h1)  
    u2, _, _ = torch.svd(cov_h2)  

    loss = 1 - torch.cosine_similarity(u1[:, 0], u2[:, 0], dim=0).mean()

    return loss
    
def latent_alignment_loss(h1, h2):
    
    alignment_loss = mse_similarity_loss(h1, h2)
    
    return alignment_loss


    

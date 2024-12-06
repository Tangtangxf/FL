import numpy as np
import pandas as pd
import pickle
from math import floor
from numpy.random import choice
import torch
from collections import defaultdict
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_num_cnt(args, list_dls_train):
    labels = []
    for dl in list_dls_train:
        labels_temp = []
        for data in dl:
            labels_temp += data[1].tolist()
        labels.append(labels_temp)

    num_cnt = []
    for label_ in labels:
        cnt = []
        total = len(label_)
        for num in range(10):
            cnt.append(label_.count(num))
        num_cnt.append(cnt)

    with open(f"dataset/data_partition_result/{args.dataset}_{args.partition}.pkl", "wb") as output:
        pickle.dump(num_cnt, output)
    print("Data partition result successfully saved!")

    # print num_cnt
    print("num_cnt table: ")
    num_cnt_table = pd.DataFrame(num_cnt, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    # print 100 rows completely
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(num_cnt_table)


def stratify_clients(args):
    partition_result_path = f"dataset/data_partition_result/{args.dataset}_{args.partition}.pkl"
    print("@@@ Start reading data_partition_result file：", partition_result_path, " @@@")

    m_data = []
    data = []

    with open(partition_result_path, 'rb') as f:
        while True:
            try:
                row_data = pickle.load(f)
                for m in row_data:
                    m_data.append(m)
            except EOFError:
                break

    # zero-mean normalizationof data
    for d in m_data:
        da = []
        avg = np.mean(d)
        std = np.std(d, ddof=1) # sample standard deviation
        for i in d:
            da.append((i - avg) / std)
        data.append(da)
    data = np.array(data)

    # The principal components analysis(PCA) of data dimension reduction
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)

    # Prototype Based Clustering: KMeans
    model = KMeans(n_clusters=args.strata_num)
    model.fit(data)
    pred_y = model.predict(data)
    pred_y= list(pred_y)
    result = []
    # put indexes into result
    for num in range(args.strata_num):
        one_type = []
        for index, value in enumerate(pred_y):
            if value==num:
                one_type.append(index)
        result.append(one_type)
    print(result)
    save_path = f'dataset/stratify_result/{args.dataset}_{args.partition}.pkl'
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as output:
        pickle.dump(result, output)

    # print silhouette_score
    s_score = metrics.silhouette_score(data, pred_y, sample_size=len(data), metric='euclidean')
    print("strata_num：", args.strata_num, " silhouette_score：", s_score, "\n")
    # silhouette score ranges from -1 to 1, higher values indicate better-defined clusters
    return result


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"saved_exp_info/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def sample_clients_without_allocation(chosen_p, choice_num):
    n_clients = len(chosen_p[0])
    strata_num = len(chosen_p)

    sampled_clients = np.zeros(len(chosen_p) * choice_num, dtype=int)

    for k in range(strata_num):
        c = choice(n_clients, choice_num, replace=False, p=chosen_p[k])
        for n_th, one_choice in enumerate(c):
            sampled_clients[k * choice_num + n_th] = int(one_choice)

    return sampled_clients


def sample_clients_with_allocation(chosen_p, allocation_number):
    n_clients = len(chosen_p[0])

    sampled_clients = []

    for i, n in enumerate(allocation_number):
        if n == 0:
            pass
        else:
            c = choice(n_clients, n, replace=False, p=chosen_p[i])
            for n_th, one_choice in enumerate(c):
                sampled_clients.append(int(one_choice))

    return sampled_clients


def cal_allocation_number(partition_result, stratify_result, sample_ratio):
    cohesion_list = []
    for row_strata in stratify_result:
        dist = np.zeros(len(row_strata))

        for j in range(len(row_strata)):
            for k in range(len(row_strata)):
                if k == j:
                    pass
                else:
                    dist[j] += np.sqrt(np.sum(np.square(np.array(partition_result[j]) - np.array(partition_result[k]))))
                    #sum of Euclidean distances between each client and all other clients in the same stratum
        dist /= len(row_strata) # each row is a stratum

        cohesion_list.append(dist)

    avg_cohesion = np.zeros(len(cohesion_list))

    for i, strata_cohesion in enumerate(cohesion_list):
        avg_cohesion[i] = sum(strata_cohesion) / len(strata_cohesion)

    allocation_number = np.zeros(len(avg_cohesion))
    for i, strata_coh in enumerate(avg_cohesion):
        weight = strata_coh / sum(avg_cohesion)
        allocation_number[i] = floor(sample_ratio * 100 * weight)

    allocation_number = allocation_number.astype(int)

    zero_num = (allocation_number == 0).sum()
    i = 0
    while np.sum(allocation_number) < sample_ratio * 100:
        if allocation_number[i] == 0:
            allocation_number[i] += max(1, int(round((sample_ratio * 100 - np.sum(allocation_number)) / zero_num)))
        i += 1

    return allocation_number

class Estimator:
    def __init__(self,train_users,alpha,M):
        self.M = M
        self.alpha = alpha
        self.train_users = train_users
        
    def query(self,userid):
        fake_response = np.random.randint(1,self.M)
        real_response = min(len(self.train_users[userid]), self.M - 1)
        #real_response = len(self.train_users[userid])
        choice = np.random.binomial(n=1,p=self.alpha)
        response = choice*real_response + (1-choice)*fake_response
        return response
    
    def estimate(self,):
        R = 0
        for uid in range(len(self.train_users)):
            R += self.query(uid)
        hat_N =  (R-len(self.train_users)*(1-self.alpha)*self.M/2)/self.alpha
        hat_N = max(hat_N,len(self.train_users))
        return hat_N
    
def local_data_sampling(dataset,K_desired,hatN):
   
    psample = K_desired/hatN
    sampled_features = []
    sampled_labels = []
    for features, labels in dataset:
        
        sample_mask = np.random.binomial(n=1, p=psample, size=len(features))
        
       
        selected_features = features[sample_mask == 1]
        selected_labels = labels[sample_mask == 1]
        
        if len(selected_features) > 0:
            sampled_features.append(selected_features)
            sampled_labels.append(selected_labels)
    
    
    if sampled_features:
        sampled_features = torch.cat(sampled_features)
        sampled_labels = torch.cat(sampled_labels)
        return sampled_features, sampled_labels
    else:
        return None, None
  
# Algorithm 1 IS 
def information_squeeze(gradient_tensor, d_prime):
    """
    Algorithm 1: Information Squeeze
    Compresses gradients by grouping similar components
    
    Args:
        gradient_tensor: Original gradient
        d_prime: Target dimension for compression
    Returns:
        compressed_gradient, group_indices
    """
    # Convert gradient tensor to numpy for clustering
    gradient_np = gradient_tensor.cpu().detach().numpy()
    
    # Initialize clustering
    kmeans = KMeans(n_clusters=d_prime, random_state=0)
    group_indices = kmeans.fit_predict(gradient_np.reshape(-1, 1))
    
    # Calculate centroids
    compressed_gradient = torch.zeros(d_prime)
    for i in range(d_prime):
        mask = (group_indices == i)
        if mask.any():
            compressed_gradient[i] = torch.tensor(gradient_np[mask].mean())
    
    return compressed_gradient, group_indices

def restore_gradient(compressed_gradient, group_indices, original_shape):
    """
    Restores original gradient shape from compressed form
    """
    restored = torch.zeros(original_shape)
    for i, idx in enumerate(group_indices):
        restored.flat[i] = compressed_gradient[idx]
    return restored

# Algorithm 2 ClientStratification
def client_stratification(compressed_gradients, num_strata):
    """
    Algorithm 2: Client Stratification
    Groups clients into strata based on their compressed gradients
    
    Args:
        compressed_gradients: List of compressed gradients from clients
        num_strata: Number of strata to create
    Returns:
        List of strata (each containing client indices)
    """
    # Convert to numpy array for clustering
    gradients_np = np.array([g.cpu().detach().numpy() for g in compressed_gradients])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_strata, random_state=0)
    strata_assignments = kmeans.fit_predict(gradients_np)
    
    # Group clients by strata
    strata = defaultdict(list)
    for client_idx, strata_idx in enumerate(strata_assignments):
        strata[strata_idx].append(client_idx)
    
    return [strata[i] for i in range(num_strata)]

def calculate_stratum_variance(gradients, stratum_indices):
    """
    Calculates the variance within a stratum
    """
    if not stratum_indices:
        return 0.0
    
    stratum_grads = [gradients[i] for i in stratum_indices]
    mean_grad = sum(stratum_grads) / len(stratum_grads)
    variance = sum((g - mean_grad).pow(2).sum() for g in stratum_grads)
    return variance.item() / len(stratum_indices)

def calculate_allocation(strata, gradients, total_samples):
    """
    Calculates number of clients to sample from each stratum
    based on size and variance
    """
    # Calculate variances and sizes
    variances = [calculate_stratum_variance(gradients, stratum) for stratum in strata]
    sizes = [len(stratum) for stratum in strata]
    
    # Calculate allocation weights
    weights = [size * var for size, var in zip(sizes, variances)]
    total_weight = sum(weights)
    
    if total_weight == 0:
        return [total_samples // len(strata)] * len(strata)
    
    # Allocate samples proportionally
    allocations = [int(total_samples * w / total_weight) for w in weights]
    
    # Distribute any remaining samples
    remaining = total_samples - sum(allocations)
    for i in range(remaining):
        allocations[i] += 1
    
    return allocations
import numpy as np
import pandas as pd
from csv import writer
import random
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from multiprocessing.pool import Pool
import warnings
warnings.filterwarnings('ignore')

count_dict = {0:(0,0), 1:(0,1), 2:(0,2), 10:(1,0), 11:(1,1), 12:(1,2),20:(2,0),21:(2,1),22:(2,2),30:(3,0),31:(3,1),32:(3,2)}

def filter_data(pitcher, pitch_type, xHB):
    all_pitches = pd.read_csv("FullPitchData2023.csv")
    filtered_pitches = all_pitches[(all_pitches["pitch_type"] == pitch_type) & 
                                  (all_pitches["player_name"] == pitcher) &
                      (all_pitches["stand"] == xHB)]
    return filtered_pitches

def filter_count_data_from_df(filtered_pitches, count_code):
    early_counts = filtered_pitches[(filtered_pitches["balls"] == count_dict[count_code][0]) & (filtered_pitches["strikes"] == count_dict[count_code][1])]
    return early_counts

def cut_columns(df):
    return df[["plate_x","plate_z"]]

def remove_outliers(data):
    x_mu = data["plate_x"].mean()
    x_percentiles = data["plate_x"].quantile([0.25, 0.75])
    x_IQR = x_percentiles[0.75] - x_percentiles[0.25]
    y_mu = data["plate_z"].mean()
    y_percentiles = data["plate_z"].quantile([0.25, 0.75])
    y_IQR = y_percentiles[0.75] - y_percentiles[0.25]
    
    remove_waste = data[(data["plate_x"] >= (x_mu - 1.5*x_IQR)) & (data["plate_x"] <= (x_mu + 1.5*x_IQR)) & (data["plate_z"] >= (y_mu - 1.5*y_IQR)) & (data["plate_z"] <= (y_mu + 1.5*y_IQR))]
    return remove_waste

def log_like(x, pi, mu, cov):
    val = 0
    for p in range(len(pi)):
        val += np.log(pi[p]*multivariate_normal.pdf(x, mean=mu[p], cov=cov[p], allow_singular=True))
    return -val

def mean_log_like(data, pi, mu, cov):
    rows = data.shape[0]
    ll = np.zeros(rows)
    for i, row in data.iterrows():
        ll[i] = log_like(row, pi, mu, cov)

    return np.mean(ll)

def create_train(dfs, i):
    train_dfs = []
    for k in range(len(dfs)):
        if(k != i):
            train_dfs.append(dfs[k])
    return pd.concat(train_dfs)

def control_metric(data, pis, mus, variances):
    metric = 0
    for index, row in data.iterrows():
        n = len(mus)
        likelihoods = np.zeros(n)
        e_distances = np.zeros(n)
        for cluster in range(n):
            likelihoods[cluster] = pis[cluster]*multivariate_normal.pdf([row["plate_x"],row["plate_z"]], mean=mus[cluster], cov=variances[cluster], allow_singular=True)
            e_distances[cluster] = np.sqrt((mus[cluster][0] - row["plate_x"])**2 + (mus[cluster][1] - row["plate_z"])**2)
        likelihoods = likelihoods / np.sum(likelihoods)
        metric += np.dot(likelihoods, e_distances)
    return metric / data.shape[0]

def k_fold_cross_validation(data):
    shuffled = data.sample(frac=1)
    split_dfs = np.array_split(shuffled, 5)

    #can't fit gmm with train data of single data point
    if (data.shape[0] < 10):
        return 1
    
    optimal_ks = np.zeros(5)
    for i in range(len(split_dfs)):
        train = create_train(split_dfs, i)
        n_train = train.shape[0]
        test_ll = np.zeros(min(n_train, 6))
        
        for j in range(1, min(n_train, 6)+1):
            fitted_gmm = GaussianMixture(n_components = j, covariance_type = "full", tol = 1e-8, max_iter=100).fit(train)
            test_ll[j-1] = -fitted_gmm.score(split_dfs[i])
   
        optimal_ks[i] = np.argmin(test_ll) + 1
    return int(np.rint(np.mean(optimal_ks)))

def map_metric_to_dist(distributions, metrics):
    metric_to_dist = {}
    for i in range(len(metrics)):
        metric_to_dist[metrics[i]] = distributions[i]
    return metric_to_dist

def gen_random_mus(k, seed):
    np.random.seed(seed)
    mus = []
    for i in range(k):
        mus.append([np.random.uniform(-1, 1), np.random.uniform(1.5, 4)])
    return mus

def optimal_pseudosample_weight(cm, k, train_early_count, validation_early_count):
    step_size = 0.05
    weight_partition = np.arange(step_size,1+step_size,step_size)
    early_count_gmm = GaussianMixture(n_components = k, covariance_type = "full", tol = 1e-8, max_iter=500, n_init=5).fit(train_early_count)
    min_ll = mean_log_like(validation_early_count, early_count_gmm.weights_, early_count_gmm.means_, early_count_gmm.covariances_)
    min_weight = 0
    for w in weight_partition:
        pis, mus, covs, ll = rr_npl(cm, train_early_count, k, w)
        validation_ll = mean_log_like(validation_early_count, pis, mus, covs)
        
        if validation_ll < min_ll:
            min_ll = validation_ll
            min_weight = w
    return min_weight

def verify_invertible_covs(covs):
    for c in covs:
        while(np.any(np.linalg.eigvals(c) <= 0)):
            c += 1e-5 * np.eye(2)
    return covs

def em_with_weights(samples, weights, k, seed):
    max_iter = 100
    tol = 1e-8
    ll_old = 0

    np.random.seed(seed)
    pis = np.random.dirichlet(np.ones(k), 1)[0]
    mus = gen_random_mus(k, seed)
    covs = [np.eye(2)] * k
    
    n = samples.shape[0]

    for j in range(max_iter):       
        #e step
        etas = np.zeros((n, k))
        for index, row in samples.iterrows():
            p_hat = np.zeros(k)
            for cluster in range(k):
                p_hat[cluster] = multivariate_normal.pdf(row, mean=mus[cluster], cov=(covs[cluster] / weights[index]), allow_singular=True)
            mixture_ps = (pis * p_hat)
            etas[index] = mixture_ps / np.sum(mixture_ps)

        #m step

        #compute pis
        for cluster in range(k):
            pis[cluster] = np.sum(etas[:, cluster]) / n

        #compute mus
        for cluster in range(k):
            numerator = np.zeros(2)
            for index, row in samples.iterrows():
                numerator += row.to_numpy() * weights[index] * etas[index][cluster]
            mus[cluster] = numerator / np.sum(weights * etas[:, cluster])

        #compute covs
        for cluster in range(k):
            numerator = np.zeros((2,2))
            for index, row in samples.iterrows():
                row = row.to_numpy()
                numerator += weights[index] * etas[index][cluster] * ((np.reshape(row, (2,1)) - np.reshape(mus[cluster], (2,1))) * (row - mus[cluster]))
            covs[cluster] = numerator / np.sum(etas[:, cluster])

        covs = verify_invertible_covs(covs)

        #check if converged sufficiently
        ll_new = mean_log_like(samples, pis, mus, covs)
        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new
    return pis, mus, covs, ll_new

def rr_npl(centering_measure, samples, k, w, seed):
    T = 250
    np.random.seed(seed)

    pseudo_samples = centering_measure.sample(n_samples=T)[0]
    
    weights = np.append(np.ones(samples.shape[0]), w*np.ones(T))
    
    samples = pd.concat([samples, pd.DataFrame(pseudo_samples, columns=samples.columns)], ignore_index=True)
    return em_with_weights(samples, weights, k, seed)

def task(R, w, k, cm, train_early_count, validation_early_count, full_count_data):
    local_seeds = random.sample(range(1000000), R)
    curr_min_val = np.inf
    min_pis = []
    min_mus = []
    min_covs = []
    for r in range(R):
        if w == 0:
            fitted_gmm = GaussianMixture(n_components = k, covariance_type = "full", tol = 1e-8, max_iter=100, init_params="random_from_data", random_state=local_seeds[r]).fit(train_early_count)
            mll = mean_log_like(train_early_count, fitted_gmm.weights_, fitted_gmm.means_, fitted_gmm.covariances_)
            if (mll < curr_min_val):
                min_pis = fitted_gmm.weights_
                min_mus = fitted_gmm.means_
                min_covs = fitted_gmm.covariances_
                curr_min_val = mll
        else:
            pis, mus, covs, ll = rr_npl(cm, train_early_count, k, w, local_seeds[r])
            validation_ll = mean_log_like(validation_early_count, pis, mus, covs)
            
            if (validation_ll < curr_min_val):
                min_pis = pis
                min_mus = mus
                min_covs = covs
                curr_min_val = validation_ll
    
    print("Task Done", flush=True)
    print(min_pis, min_mus, min_covs, flush=True)
    min_distribution = {"PI":min_pis,"MU":min_mus,"COV":min_covs}

    return min_distribution

def weights_task(weight, cm, k, train_early_count, validation_early_count, seeds, i):
    if weight == 0:
        early_count_gmm = GaussianMixture(n_components = k, covariance_type = "full", tol = 1e-8, max_iter=500, n_init=5).fit(train_early_count)
        return mean_log_like(validation_early_count, early_count_gmm.weights_, early_count_gmm.means_, early_count_gmm.covariances_)
    else:
        pis, mus, covs, ll = rr_npl(cm, train_early_count, k, weight, seeds[i])
        return mean_log_like(validation_early_count, pis, mus, covs)

def metrics_task(full_count_data, dist_dict):
    return 12 * control_metric(full_count_data, dist_dict["PI"], dist_dict["MU"], dist_dict["COV"])
    
#single function to call with pitcher, pitch, xHB, count_code that orchestrates the rest
def control(pitcher, pitch, xHB, count_code):    
    all_data = filter_data(pitcher, pitch, xHB)
    all_data = remove_outliers(all_data)
    
    count_data = filter_count_data_from_df(all_data, count_code)
    count_data.reset_index(inplace=True,drop=True)
    count_data = cut_columns(count_data)

    train_total, validation_total = train_test_split(all_data,test_size=0.3)

    train_early_count = filter_count_data_from_df(train_total, count_code)
    validation_early_count = filter_count_data_from_df(validation_total, count_code)

    all_data = cut_columns(all_data)
    train_total = cut_columns(train_total)
    train_early_count = cut_columns(train_early_count)
    validation_early_count = cut_columns(validation_early_count)
    validation_early_count.reset_index(inplace=True,drop=True)
    train_early_count.reset_index(inplace=True,drop=True)
    
    #get optimal k from V filtered by count code
    k = k_fold_cross_validation(count_data)

    #cm used to get train to get w (need out of sample loss) fitted_gmm is overall centering measure
    #use gmm with all data as centering measure when there isn't enough specific count data
    if (train_total.shape[0] == 1 or train_total.shape[0] <= k):
        cm = GaussianMixture(n_components = k, covariance_type = "full", tol = 1e-8, max_iter=500, n_init=5).fit(all_data)
    else:
        cm = GaussianMixture(n_components = k, covariance_type = "full", tol = 1e-8, max_iter=500, n_init=5).fit(train_total)
    fitted_gmm = GaussianMixture(n_components = k, covariance_type = "full", tol = 1e-8, max_iter=500, n_init=5).fit(all_data)

    step_size = 0.05
    weight_partition = np.arange(step_size,1+step_size,step_size)

    print("Starting weight partition")
    seeds = random.sample(range(1000), 100)
    
    pool = Pool()
    weight_lls = [pool.apply_async(weights_task, args=(j, cm, k, train_early_count, validation_early_count,seeds, int(count),)).get() for count, j in enumerate(weight_partition)]
    pool.close()
    pool.join()
    
    min_ll = min(weight_lls)
    w = weight_lls.index(min_ll) * step_size

    print("Chose weight")
    
    B = 100
    R = 5
    
    pool = Pool()
    distributions = [pool.apply_async(task, args=(R, w, k, fitted_gmm, train_early_count, validation_early_count, count_data,)).get() for i in range(B)]
    pool.close()
    pool.join()

    pool = Pool()
    metrics = [pool.apply_async(metrics_task, args=(count_data, distributions[k],)).get() for k in range(B)]
    pool.close()
    pool.join()

    dist_to_metric = map_metric_to_dist(distributions, metrics)
    
    #sort metrics in decreasing order
    metrics.sort()

    median_dist = dist_to_metric[metrics[49]]
    all_pitch_w_outliers = filter_data(pitcher, pitch, xHB)
    return control_metric(all_pitch_w_outliers, median_dist["PI"],median_dist["MU"],median_dist["COV"]), median_dist

if __name__ == "__main__":
    median, med_distribution = control("Gore, MacKenzie", "FF", "L", 21)
    result = ["Gore, MacKenzie",median,med_distribution["PI"],med_distribution["MU"],med_distribution["COV"]]
    with open('FB-2023-21Count-LHB.csv', 'a') as filestream:
        writer_object = writer(filestream)
        writer_object.writerow(result)
        filestream.close()
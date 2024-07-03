import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


def filter_data(pitch_df, pitcher):
    filtered_df = pitch_df[pitch_df["player_name"] == pitcher]
    filtered_df = filtered_df[["plate_x","plate_z"]]
    return filtered_df

#remove outliers based on 1.5*IQR in x and y axes
def remove_outliers(data):
    x_mu = data["plate_x"].mean()
    x_percentiles = data["plate_x"].quantile([0.25, 0.75])
    x_IQR = x_percentiles[0.75] - x_percentiles[0.25]
    y_mu = data["plate_z"].mean()
    y_percentiles = data["plate_z"].quantile([0.25, 0.75])
    y_IQR = y_percentiles[0.75] - y_percentiles[0.25]
    
    remove_waste = data[(data["plate_x"] >= (x_mu - 1.5*x_IQR)) & (data["plate_x"] <= x_mu + 1.5*x_IQR) & (data["plate_z"] >= y_mu - 1.5*y_IQR) & (data["plate_z"] <= y_mu + 1.5*y_IQR)]
    return remove_waste

#qualified if >=250 pitches thrown, easy to change threshold here
def qualified_pitchers(pitch_df):
    qp = []
    for p in pitch_df.player_name.unique():
        if (pitch_df[pitch_df["player_name"] == p].shape[0] >= 250):
            qp.append(p)    
    return qp

def fit_em(data, num_clusters):    
    em = GaussianMixture(n_components = num_clusters, covariance_type = "full", tol = 1e-8, max_iter=100).fit(data)
    return em

def optimal_k(train, test):  
    log_likelihoods = np.zeros(6)   
    #compute log likelihood for each k (1, 2, 3, 4, 5, 6)
    for i in range(1,7):
        fitted_gmm = fit_em(train, i)
        log_likelihoods[i-1] = -fitted_gmm.score(test)
   
    return np.argmin(log_likelihoods) + 1

def bootstrapped_df(df):
    indices = np.random.randint(df.shape[0], size = df.shape[0])
    return df.iloc[indices,:]

def likelihood(pi, x, y, mu, var):
    return pi * mvn.pdf([x, y], mean=mu, cov=var)

def gmm_density(gmm, x, z):
    like = 0
    for i in range(len(gmm.weights_)):
        like += gmm.weights_[i] * mvn.pdf([x, z], mean=gmm.means_[i], cov=gmm.covariances_[i])
    return like

def control_metric(data, pis, mus, variances):
    metric = 0
    for index, row in data.iterrows():
        n = len(mus)
        likelihoods = np.zeros(n)
        e_distances = np.zeros(n)
        for cluster in range(n):
            likelihoods[cluster] = likelihood(pis[cluster], row["plate_x"], row["plate_z"], mus[cluster], variances[cluster])
            e_distances[cluster] = np.sqrt((mus[cluster][0] - row["plate_x"])**2 + (mus[cluster][1] - row["plate_z"])**2)
        likelihoods = likelihoods / np.sum(likelihoods)
        metric += np.dot(likelihoods, e_distances)
    return metric / data.shape[0]

def construct_confidence_interval(df):   
    control_metrics = np.zeros(20)
    gmm_dict = {}
    
    for n in range(20):       
        train, test = train_test_split(df, test_size=0.2)
        k = optimal_k(train, test)
        fitted_gmm = fit_em(bootstrapped_df(df), k)
        control_value = control_metric(df, fitted_gmm.weights_, fitted_gmm.means_, fitted_gmm.covariances_)
        control_metrics[n] = control_value
        gmm_dict[control_value] = fitted_gmm
        
    control_metrics = np.sort(control_metrics)
    return control_metrics[1], control_metrics[18], control_metrics[9], gmm_dict[control_metrics[9]]
    
def single_ranking(total_df, pitch, hand):
    pitch_df = total_df[(total_df["pitch_type"] == pitch) & (total_df["stand"] == hand)]
    qp = qualified_pitchers(pitch_df)
    ninetyFive = []
    five = []
    median = []
    pi = []
    mu = []
    cov = []

    for pitcher in qp:
        pitcher_df = filter_data(pitch_df, pitcher)
        pitcher_df = remove_outliers(pitcher_df)
        pitcher_df.reset_index(inplace=True,drop=True)

        lower, upper, middle, median_gmm = construct_confidence_interval(pitcher_df)

        ninetyFive.append(lower * 12)
        five.append(upper * 12)
        median.append(middle * 12)
        pi.append(median_gmm.weights_)
        mu.append(median_gmm.means_)
        cov.append(median_gmm.covariances_)

    player_value = {"Pitcher":qp, "NinetyFive":ninetyFive, "Five":five, "Median Control":median, "Pis":pi, "Mus":mu, "Covariances":cov}
    ranking = pd.DataFrame(player_value)
    ranking.sort_values(by="Median Control", ascending=True, inplace=True)
    ranking.to_csv(str(pitch) + "-" + str(hand) + "HB-2023.csv")

#create ranking of control for every pitch type in 2023
def create_rankings():
    total_df = pd.read_csv("FullPitchData2023.csv")
    total_df = total_df.dropna(subset=["pitch_type"])
    pitch_types = total_df["pitch_type"].unique()
    handedness = ["R","L"]
    
    for pitch in pitch_types:
        for hand in handedness:
            single_ranking(total_df, pitch, hand)
            
#compute control value for particular pitcher, pitch, handedness of batter
def specific_control(pitcher, pitch, hand):
    total_df = pd.read_csv("FullPitchData2023.csv")
    filtered_df = total_df[(total_df["player_name"] == pitcher) & (total_df["pitch_type"] == pitch) & (total_df["stand"] == hand)]
    filtered_df = filtered_df[["plate_x","plate_z"]]
    filtered_df = remove_outliers(filtered_df)
    B = 100
    dist_dict = {}
    controls = np.zeros(B)
    for i in range(B):
        train, test = train_test_split(filtered_df, test_size=0.2)
        k = optimal_k(train, test)
        fitted_gmm = fit_em(bootstrapped_df(filtered_df), k)
        c_value = control_metric(filtered_df, fitted_gmm.weights_, fitted_gmm.means_, fitted_gmm.covariances_)
        controls[i] = c_value
        dist_dict[c_value] = fitted_gmm
    return controls[49], dist_dict[controls[49]]

#visualize the distribution, change name of figure to save as on last line
def true_density(fitted_gmm):
    grid = [(x/100, y/100) for x in range(-150, 150) for y in range(150, 450)]
    weights = np.zeros(len(grid))
    x = np.zeros(len(grid))
    z = np.zeros(len(grid))
    for w in range(len(grid)):
        weights[w] = gmm_density(fitted_gmm, grid[w][0], grid[w][1])
        x[w] = grid[w][0]
        z[w] = grid[w][1]
    
    pitches = pd.DataFrame({"x":x, "z":z, "density":weights})
    kde = sns.kdeplot(data=pitches, x="x", y="z", weights = "density", fill=True, thresh=0, levels=100, cmap="mako")
    fig = kde.get_figure()
    fig.savefig("KirbyRHBFF2023.png")
    
#get overall distribution for 1-0 fastballs against RHB
def overall_specific_case(pitch, b, s, hand):
    total_df = pd.read_csv("FullPitchData2023.csv")
    filtered_df = total_df[(total_df["balls"] == b) & (total_df["strikes"] == s) & (total_df["pitch_type"] == pitch) & (total_df["stand"] == hand)]
    filtered_df = filtered_df[["plate_x","plate_z"]]
    filtered_df = remove_outliers(filtered_df)
    train, test = train_test_split(filtered_df, test_size=0.2)
    k = optimal_k(train, test)
    fitted_gmm = fit_em(bootstrapped_df(filtered_df), k)
    true_density(fitted_gmm)

#check number of pitches thrown in this case to decide if stable
def num_pitches_case(pitcher, pitch, b, s, hand):
    total_df = pd.read_csv("FullPitchData2023.csv")
    filtered_df = total_df[(total_df["player_name"] == pitcher) & (total_df["balls"] == b) & (total_df["strikes"] == s) & (total_df["pitch_type"] == pitch) & (total_df["stand"] == hand)]
    filtered_df = filtered_df[["plate_x","plate_z"]]
    filtered_df = remove_outliers(filtered_df)
    print(filtered_df.shape[0])
    
#visualization in particular case. only use for visualization, likely wont be stable due to low n
def pitcher_specific_case(pitcher, pitch, b, s, hand):
    total_df = pd.read_csv("FullPitchData2023.csv")
    filtered_df = total_df[(total_df["player_name"] == pitcher) & (total_df["balls"] == b) & (total_df["strikes"] == s) & (total_df["pitch_type"] == pitch) & (total_df["stand"] == hand)]
    filtered_df = filtered_df[["plate_x","plate_z"]]
    filtered_df = remove_outliers(filtered_df)
    train, test = train_test_split(filtered_df, test_size=0.2)
    k = optimal_k(train, test)
    fitted_gmm = fit_em(bootstrapped_df(filtered_df), k)
    true_density(fitted_gmm)
    
#average control for a pitch type between all qualified pitchers over LHB and RHB for unified number
def best_overall_control(pitch):
    left = "-LHB-2023.csv"
    right = "-RHB-2023.csv"
    left_df = pd.read_csv(left)
    right_df = pd.read_csv(right)
    left_pitchers = left_df["Pitcher"].unique()
    right_pitchers = right_df["Pitcher"].unique()
    pitchers = list(set(left_pitchers) & set(right_pitchers))
    avg_control = []
    for p in pitchers:
        avg_control.append((left_df[left_df["Pitcher"] == p].values[0][4] + right_df[right_df["Pitcher"] == p].values[0][4])/2)
    controls = {"Pitcher":pitchers, "Control":avg_control}
    ranking = pd.DataFrame(controls)
    ranking.sort_values(by="Control", ascending=True, inplace=True)
    filename = pitch + "-Overall-2023.csv"
    ranking.to_csv(filename)
    return ranking
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn

#refer to notes in journal for understanding and this link for coding the algorithm (though it needs to be changed for 2 bivariate normal distributions) 
def filter_data(pitcher, pitch_type, xHB, count):
    all_pitches = pd.read_csv("FullPitchData2022.csv")
    filtered_pitches = all_pitches[(all_pitches["pitch_type"] == pitch_type) & 
                                  (all_pitches["player_name"] == pitcher) &
                      (all_pitches["stand"] == xHB)]

    if count == 3:
        filtered_pitches = filtered_pitches[filtered_pitches["balls"] == 3]
    elif count == 2:
        print("2 Strike Count")
        filtered_pitches = filtered_pitches[filtered_pitches["strikes"] == 2]

    filtered_pitches = filtered_pitches[["plate_x","plate_z"]]
    return filtered_pitches

def remove_outliers(data):
    x_mu = data["plate_x"].mean()
    x_percentiles = data["plate_x"].quantile([0.25, 0.75])
    x_IQR = x_percentiles[0.75] - x_percentiles[0.25]
    y_mu = data["plate_z"].mean()
    y_percentiles = data["plate_z"].quantile([0.25, 0.75])
    y_IQR = y_percentiles[0.75] - y_percentiles[0.25]
    
    remove_waste = data[(data["plate_x"] >= (x_mu - 1.5*x_IQR)) & (data["plate_x"] <= x_mu + 1.5*x_IQR) & (data["plate_z"] >= y_mu - 1.5*y_IQR) & (data["plate_z"] <= y_mu + 1.5*y_IQR)]
    return remove_waste

def likelihood(x, y, mu, var):
    return mvn.pdf([x, y], mean=mu, cov=var)

def control_metric(data, mus, variances):
    metric = 0
    for index, row in data.iterrows():
        n = len(mus)
        likelihoods = np.zeros(n)
        e_distances = np.zeros(n)
        for cluster in range(n):
            likelihoods[cluster] = likelihood(row["plate_x"], row["plate_z"], mus[cluster], variances[cluster])
            e_distances[cluster] = np.sqrt((mus[cluster][0] - row["plate_x"])**2 + (mus[cluster][1] - row["plate_z"])**2)
        likelihoods = likelihoods / np.sum(likelihoods)
        metric += np.dot(likelihoods, e_distances)
    return metric / data.shape[0]

def qualified_pitchers(pitch_type, n):
    qualified = []
    data = pd.read_csv("FullPitchData2022.csv")
    for p in data.player_name.unique():
        if (data[(data["player_name"] == p) & (data["pitch_type"] == pitch_type)].shape[0] >= n):
            qualified.append(p)
    
    return qualified

def create_ranking(num_clusters, hand_filter, count, pitch_type, qualified_pitches):
    pitchers = qualified_pitchers(pitch_type, qualified_pitches)
    values = []
    saved_mus = [[]] * num_clusters
    saved_variances = [[]] * num_clusters
    saved_pis = []
    for p in pitchers:
        val = 0
        for hand in hand_filter:
            print(p, hand)
            df = filter_data(p, pitch_type, hand, count)
            df = remove_outliers(df)
            pi, mus, variances = em(df, num_clusters)
            measure = control_metric(df, mus, variances)
            for i in range(len(mus)):
                #multiplication by 12 to convert unit from feet to inches
                if saved_mus[i] == []:
                    saved_mus[i] = [12 * np.array(mus[i])]
                    saved_variances[i] = [12 * np.array(variances[i])]
                else:
                    saved_mus[i].append(12 * np.array(mus[i]))
                    saved_variances[i].append(12 * np.array(variances[i]))
            saved_pis.append(pi)
            val += measure
        values.append(12 * (val / len(hand_filter)))

    saved_pis = np.array(saved_pis)
    saved_pis = np.transpose(saved_pis)
    
    print("Values in order: ", values)
    print("Pitchers in order: ", pitchers)
    player_value = {"Pitcher":pitchers, "Control":values}
    ranking = pd.DataFrame(player_value)
    
    #don't add distribution information in csv when considering LHB and RHB (clutters it, keep that info in files specifically against one handedness of batter)
    if (hand_filter != ["L", "R"]):
        for j in range(num_clusters):
            cluster_num = str(j+1)
            ranking["Pi " + cluster_num] = saved_pis[j]
            ranking["Mu " + cluster_num] = saved_mus[j]
            ranking["Var " + cluster_num] = saved_variances[j]

    ranking.sort_values(by="Control", ascending=True, inplace=True)    
    ranking.to_csv("TwoClustersLHB-FB-2022.csv")
    return ranking

def visualize_cluster(mu1, v1, mu2, v2):
    a, b = np.mgrid[-1:1:.01, 1.5:5:.01]
    pos = np.dstack((a, b))
    rv = mvn(mu1, v1)
    rv2 = mvn(mu2, v2)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax2.contourf(a, b, rv.pdf(pos))
    ax3.contourf(a, b, rv2.pdf(pos))

#https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html
def em(data, num_clusters):
    max_iter = 100
    tolerance = 0.01

    #mus are roughly the 4 corners + "middle" of the strikezone
    mus = [[0, 2.5], [-0.5, 2.6], [0.5, 2.4], [-0.5, 2.4], [0.5, 2.6]]
    mus = mus[:num_clusters]
    init_var = [[0.544085, -0.284055],[-0.284055, 0.669853]]
    variances = [init_var] * num_clusters

    pi = [1/num_clusters] * num_clusters
    num_variables = 2
    num_data_points = data.shape[0]
    likelihoods = []
    
    old_likelihood = 0
    for i in range(max_iter):
        
        #E-Step: create weighting that each data point in each cluster (then normalize)
        weights = np.zeros((num_clusters, num_data_points))
        for j in range(num_clusters):
            for i in range(num_data_points):
                weights[j, i] = pi[j] * mvn(mus[j], variances[j]).pdf(data.iloc[i].dropna().to_numpy())
        weights /= weights.sum(0)
        
        #M-Step
        pi = np.zeros(num_clusters)
        for j in range(num_clusters):
            for i in range(num_data_points):
                pi[j] += weights[j, i]
        pi /= num_data_points
        
        mus = np.zeros((num_clusters, num_variables))
        for j in range(num_clusters):
            for i in range(num_data_points):
                mus[j] += weights[j, i] * data.iloc[i].dropna().to_numpy()
            mus[j] /= weights[j, :].sum()
        
        variances = np.zeros((num_clusters, num_variables, num_variables))
        for j in range(num_clusters):
            for i in range(num_data_points):
                ys = np.reshape(data.iloc[i].dropna().to_numpy() - mus[j], (2,1))
                variances[j] += weights[j, i] * np.dot(ys, ys.T)
            variances[j] /= weights[j,:].sum()

        #compute likelihood
        new_likelihood = 0.0
        for i in range(num_data_points):
            s = 0
            for j in range(num_clusters):
                s += pi[j] * mvn(mus[j], variances[j]).pdf(data.iloc[i].dropna().to_numpy())
            new_likelihood += np.log(s)
        likelihoods.append(new_likelihood)

        if (np.abs(old_likelihood - new_likelihood) < tolerance):
            break       
        old_likelihood = new_likelihood
        
    #graph the log-likelihoods over the iterations
    #plt.plot(range(len(likelihoods)), likelihoods, '-o')
    #plt.xlabel("Iteration Number")
    #plt.ylabel("Log Likelihood")
    #plt.show()

    return pi, mus, variances

def squared_error(x, y, mus):
    sse = 0
    for i_means in mus:
        sse += np.sqrt((i_means[0] - x)**2 + (i_means[1] - y)**2)

    return sse/len(mus)

def optimize_cluster_count(df):
    df = df[["plate_x","plate_z"]]
    
    cluster_mse = np.zeros(5)
    '''
    trying this (https://stats.stackexchange.com/questions/465124/what-is-the-mathematical-definition-of-the-elbow-method) where also dividing by number of clusters to hopefully get a parabolic result instead of an elbow. basically (1/n) SUM [(1/num clusters) * SUM d(point, cluster i)^2]
    '''
    for cluster in range(1, 6):
        #get cluster centers
        pi, mus, variances = em(df, cluster)
        
        temp_mse = 0
        #from the results of em then get the mean squared error for each data point
        for index, row in df.iterrows():
            temp_mse += squared_error(row["plate_x"], row["plate_z"], mus)
        cluster_mse[cluster - 1] = temp_mse/df.shape[0]

    plt.plot(range(len(cluster_mse)), cluster_mse, '-o')
    plt.xlabel("Cluster Number")
    plt.ylabel("MSE")
    plt.show()

    return (np.argmin(cluster_mse) + 1)

def specific_rankings(num_clusters, hand, count, pitch_type, qualified_pitches):
    handedness = ["L","R"]
    if hand == "R":
        handedness = ["R"]
    elif hand == "L":
        handedness = ["L"]
        
    #count = -1 (dont filter), = 2 (2 strike counts), 3 (3 ball counts)
    create_ranking(num_clusters, handedness, count, pitch_type, qualified_pitches)
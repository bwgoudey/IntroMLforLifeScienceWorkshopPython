import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.stats import multivariate_normal as mvn


# Add some random simulated to a given data.frame
def append_simulated(df, n_simulated=20):
    simulated_measures = np.random.rand(df.shape[0],n_simulated)
    simulated_measures = StandardScaler().fit(simulated_measures).transform(simulated_measures)
    simulated_names = ["simulated_{}".format(i) for i in range(n_simulated)]
    simulated_df = pd.DataFrame(data=simulated_measures,columns=simulated_names)
    return(pd.concat([df,simulated_df], axis=1))

# Change the target to be binary
def binarize_y(y):     
    return(1*(y>100))

# Rename a number of columns to make them more describptive
def tidy_diabetes_names(X):     
    # We rename the variables to be more descriptive
    X.rename({'s1': 'tc', 
              's2': 'ldl', 
              's3': 'hdl',
              's4': 'tch',
              's5': 'ltg',
              's6': 'glu',}
             , axis=1, inplace=True)
    return(X)


#
#
# Generate a new dataset based on the existing diabetes dataset
def generate_novel_data(n_samples=1000, n_simulated=20):

    # Reload the diabetes dataset to retrieve the continuous progression score    
    X,y=datasets.load_diabetes(as_frame=True, return_X_y=True)
    
    # We rename the variables to be more descriptive
    X = tidy_diabetes_names(X)

    # Combine X and y into a single data.frame
    X_y = X.assign(y = y)
    
    # Based on existing data means and covariance, generate some new data 
    # (assumes mutlivariate normal, unlikely to be true but good enough)
    # Then turn into a data.frame
    sim_dat = mvn.rvs(mean = X_y.mean(), cov=X_y.cov(), size = n_samples)
    sim_X_y = pd.DataFrame(data = sim_dat, columns = X_y.columns)

    # Split back in to X and y
    sim_y = binarize_y(sim_X_y['y'])
    sim_X = sim_X_y.drop(columns=['y'])
    
    #If we've specified, add some number of randomly generated features   
    if n_simulated>0:
        sim_X=append_simulated(sim_X, n_simulated)
    
    return(sim_X, sim_y)
    
    
#
# Load in the diabetes dataset that comes with sklearn. 
# Then tidy names, binaryise the progression variable and add some simulated (if specified)
def load_diabetes_data(diabetes_df_raw, scale=True, add_n_features=0, sample_limit=1000):
        # Select columns excluding 'id' and 'year_of_followup'
    diabetes_df = diabetes_df_raw.drop(columns=['id', 'year_of_followup'])

    # Convert 'diabetes' to a categorical type
    diabetes_df['diabetes'] = diabetes_df['diabetes'].astype('category')

    # If add_n_features is greater than 0, append random features
    if add_n_features > 0:
        diabetes_df = append_simulated(diabetes_df, add_n_features)

    # Scale the data if scale is True, excluding the 'diabetes' column
    if scale:
        #print(diabetes_df)
        features_to_not_scale=['diabetes','site','gender_1_male_2_female']
        features_to_scale = diabetes_df.drop(columns=features_to_not_scale).columns
        scaler = StandardScaler()
        diabetes_df[features_to_scale] = scaler.fit_transform(diabetes_df[features_to_scale])
    
    # Set nsamples to the minimum of nsamples, number of rows in the DataFrame
    sample_limit = min(sample_limit, diabetes_df.shape[0])
    # Study 1: filter for site == 5, group by 'diabetes', and sample from each group
    diabetes_study1_df = (diabetes_df[diabetes_df['site'] == 5]
                        .groupby('diabetes')
                        .apply(lambda x: x.head(np.ceil(sample_limit / 2).astype(int)))
                        .reset_index(drop=True)
                        .drop(columns=['site']))

    # Study 2: filter for site != 5, group by 'diabetes', and drop 'site' column
    diabetes_study2_df = diabetes_df[diabetes_df['site'] != 5].drop(columns=['site'])
    X1=diabetes_study1_df.drop(columns=['diabetes'])
    y1=diabetes_study1_df['diabetes']
    X2=diabetes_study2_df.drop(columns=['diabetes'])
    y2=diabetes_study2_df['diabetes']

    # Return the dictionary
    return X1, y1, X2, y2


# Plot a ROC curve with a label
def plot_roc(y, yp, label="", ax=None, color='blue'):
    fpr, tpr, thresh = metrics.roc_curve(y, yp)
    auc = metrics.roc_auc_score(y, yp)
    if ax:
        ax.plot(fpr,tpr,color=color, label="{} AUC={:.2f}".format(label, auc))
    else:
        plt.plot(fpr,tpr,color=color, label="{} AUC={:.2f}".format(label, auc))

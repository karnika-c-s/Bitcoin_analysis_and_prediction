#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

#importing datasets from csv form
dataset = pd.read_csv("D:\DCS\Projects\Sem 4 PA Project-\Bitcoin.csv",parse_dates=True, index_col="Date", dayfirst=True)

print("Dimension of dataset:", dataset.shape)
print(dataset.info())

#sample dataset
print("Sample Dataset")
print(dataset.head())

#summary
print("Summary")
print(dataset.describe())

#null value checking
missing_values=dataset.isnull().sum()
print("Checking for missing values")
print(missing_values)

# Drop any missing values if present
dataset.dropna(inplace=True)

# Feature Selection
df=['Open','High','Low','Close','Adj Close','Volume']
data=dataset[df]

#plotting graph for a single column 
dataset['Close'].plot()
plt.title('PLot of Close value from 2018-2021')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

#plotting graph for many column using 'subplot function'
dataset.plot(subplots=True,figsize=(5,5))
plt.show()


#resampling the time series data based on monthly
dataset_month=dataset.resample("M").mean()
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(dataset_month.loc['2018'].index,dataset_month.loc['2018',"Close"],width=20,align='center')
plt.title('Plot on close values(Monthwise mean- resampling method)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

#finding the trend in the dataset for "Close" for given infos
window_size=40
rolling_mean=dataset['Close'].rolling(window_size).mean()    #using rolling mean we're finding the trend 
rolling_mean.plot()
plt.title('TREND IN CLOSE VALUE')
plt.xlabel('DATE')
plt.ylabel('CLOSE')
plt.show()

# Normalize the data using Min-Max scaling
scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(data)

# Create a new DataFrame with scaled data
scaled_dataset=pd.DataFrame(scaled_data, columns=df, index=dataset.index)

# Print the preprocessed dataset
print(scaled_dataset.head())

###PRINCIPAL COMPONENT ANALYSIS

data=dataset[['Open','Volume']]

# Normalize the data (excluding the Date column)
scalar=StandardScaler()
data_scaled=scalar.fit_transform(data)

# PCA
pca=PCA(n_components=2)
data_pca=pca.fit_transform(data_scaled)

# Calculate the principal components and their corresponding explained variances
principal_components=pca.components_
explained_variances=pca.explained_variance_ratio_
plt.bar(range(len(explained_variances)), explained_variances)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()

# Print the explained variances
print("\n\nExplained variances by principal component:")
for i, explained_variance in enumerate(explained_variances):
    print(f"Principal Component {i+1}: {explained_variance}")
 
##TIME SERIES FORECASTING

#forecasting the Volume for the next 2 years
model=ARIMA(dataset_month['Close'], order=(1, 0, 0))
model_fit=model.fit()
forecast=model_fit.forecast(steps=24)     #24 means 24 months, i.e: 2 years

last_date=dataset_month.index[-1]
forecast_dates=pd.date_range(start=last_date, periods=24, freq='M')

#forecasting plot model for 2 years (VOLUME)
plt.figure(figsize=(10,6))
plt.plot(dataset_month.index, dataset_month['Close'], label='Actual')
plt.plot(forecast_dates, forecast, label='Forecast')
plt.title("\n\nClose value Forecast for the Next 2 Years")
plt.xlabel('DATE')
plt.ylabel('CLOSE')
plt.legend()
plt.show()

# Print the forecast values
print("Close value forecast for the next 2 years:")
for date,value in zip(forecast_dates, forecast):
    print(f"{date}:{value}")

###CLUSTERING
# Determine the optimal number of clusters using silhouette score
silhouette_scores=[]
k_values=range(2,11)
n_init=10                   

for k in k_values:
    kmeans=KMeans(n_clusters=k,random_state=0,n_init=n_init)
    labels=kmeans.fit_predict(data_pca)
    silhouette_scores.append(silhouette_score(data_pca,labels))

# Plot the silhouette scores
plt.plot(k_values,silhouette_scores,marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by Number of Clusters')
plt.show()

#Outlier Analysis
sns.boxplot(data_pca)
plt.show()

data_df = pd.DataFrame(data_pca)
data_df.columns = ['Open' , 'High']
data_df['Open'] = data_df['Open'].interpolate()
data_df['High'] = data_df['High'].interpolate()

n_init=10   
optimal_k=silhouette_scores.index(max(silhouette_scores))+2
kmeans=KMeans(n_clusters=optimal_k, random_state=0)
labels=kmeans.fit_predict(data_pca)

# Add the cluster labels to the dataset
dataset['Cluster']=labels
data_df['Cluster'] = labels

# Display the cluster labels in a table
cluster_table=pd.DataFrame({'Date':dataset.index,'Cluster': labels})
cluster_inf = data_df.groupby(data_df['Cluster']).mean().reset_index()
print(cluster_inf)

# Plot the clusters
plt.scatter(data_pca[:,0], data_pca[:,1], c=labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustering Analysis')
plt.show()

# Count the number of occurrences for each cluster
cluster_counts=cluster_table['Cluster'].value_counts().reset_index()
cluster_counts.columns=['Cluster','Count']
cluster_counts=cluster_counts.sort_values('Cluster')

# Print the cluster counts table
print("\n\nCluster Counts:")
print(cluster_counts)

# Feature Selection
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
X = dataset[features]
y = dataset['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
print("Random Forest Mean Squared Error:", rf_mse)

# Random Forest Bar Plot
bar_width = 0.35
num_data_points = len(y_test)
x = np.arange(num_data_points)
plt.figure(figsize=(8, 4))
plt.bar(x, y_test.values, width=bar_width, label="Actual Price")
plt.bar(x + bar_width, rf_predictions, width=bar_width, label="Random Forest Predictions")
plt.xlabel('Data Point')
plt.ylabel('Close Price')
plt.title('Bitcoin Price Prediction - Random Forest')
plt.legend()
plt.show()


# Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_regressor.fit(X_train, y_train)
gb_predictions = gb_regressor.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_predictions)
print("Gradient Boosting Mean Squared Error:", gb_mse)

# Gradient Boosting Bar Plot
plt.figure(figsize=(8, 4))
plt.bar(x, y_test.values, width=bar_width, label="Actual Price")
plt.bar(x + bar_width, gb_predictions, width=bar_width, label="Gradient Boosting Predictions")
plt.xlabel('Data Point')
plt.ylabel('Close Price')
plt.title('Bitcoin Price Prediction - Gradient Boosting')
plt.legend()
plt.show()

# Neural Network Regressor
nn_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
nn_regressor.fit(X_train, y_train)
nn_predictions = nn_regressor.predict(X_test)
nn_mse = mean_squared_error(y_test, nn_predictions)
print("Neural Network Mean Squared Error:", nn_mse)

# Neural Network Bar Plot
plt.figure(figsize=(8, 4))
plt.bar(x, y_test.values, width=bar_width, label="Actual Price")
plt.bar(x + bar_width, nn_predictions, width=bar_width, label="Neural Network Predictions")
plt.xlabel('Data Point')
plt.ylabel('Close Price')
plt.title('Bitcoin Price Prediction - Neural Network')
plt.legend()
plt.show()

# Find the date with the highest closing price
highest_close_date = y_test.idxmax()
highest_close_price = y_test.max()

print("Highest Closing Price:", highest_close_price)
print("Date of Highest Closing Price:", highest_close_date)
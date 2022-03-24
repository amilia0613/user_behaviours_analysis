from flask import Flask, render_template
import time
import datetime
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from io import BytesIO
import base64

app = Flask(__name__)

#Objective 1: User and Session Identification
weblog_path = os.path.join("weblog.csv")
weblog = pd.read_csv(weblog_path)

weblog['date'] = weblog['datetime'].str.slice(0, 2)

weblog['request'] = weblog['request'].str.slice(10, )

main = weblog[weblog['request'] == '']
main["page_category"] = "main_page"

landing_page = weblog[weblog['request'].str.contains('landing_page')==True]
landing_page["page_category"] = "landing_page"

signup = weblog[weblog['request'].str.contains('signup')==True]
signup["page_category"] = "sign_up"

login = weblog[weblog['request'].str.contains('login')==True]
login["page_category"] = "login"

dashboard = weblog[weblog['request'].str.contains('dashboard')==True]
dashboard["page_category"] = "dashboard"

scoreboard = weblog[weblog['request'].str.contains('scoreboard')==True]
scoreboard["page_category"] = "scoreboard"

profile = weblog[weblog['request'].str.contains('profile')==True]
profile["page_category"] = "profile"

available_rewards = weblog[weblog['request'].str.contains('available_rewards')==True]
available_rewards["page_category"] = "available_rewards"

faq = weblog[weblog['request'].str.contains('faq')==True]
faq["page_category"] = "faq"

contact_us = weblog[weblog['request'].str.contains('contact_us')==True]
contact_us["page_category"] = "contact_us"

logout = weblog[weblog['request'].str.contains('logout')==True]
logout["page_category"] = "logout"

quiz = weblog[weblog['request'].str.contains('quiz')==True]
quiz["page_category"] = "quiz"

weblog_df = pd.concat([main, landing_page, signup, login, dashboard, scoreboard, profile, available_rewards, faq, contact_us, logout, quiz])

weblog_df = weblog_df.sort_values('datetime')

weblog_df['user'].nunique()
weblog_df['page_category'].nunique()

path = 'weblog_df.csv'

with open(path, 'w', encoding = 'utf-8-sig') as f:
  weblog_df.to_csv(f)


#Objective 2: Discover User Navigation Pattern
weblog_path = os.path.join("weblog_df.csv")
weblog_df = pd.read_csv(weblog_path)

page_cluster = weblog_df.groupby('page_category')['request'].count()
page_cluster = page_cluster.reset_index()
page_cluster.columns = ['page_category', 'num_of_hits']

weblog_df['time_on_page'] = ''

weblog_df['user'] = weblog_df['user'].astype('category')
weblog_df['datetime'] = pd.to_datetime(weblog_df['datetime'], format='%d/%b/%Y:%H:%M:%S %z')
weblog_df['method'] = weblog_df['method'].astype('category')

weblog_df['datetime'][3]-weblog_df['datetime'][2]

#Calculate users' time on page 
def time_on_page(dataset):
    unique_values = dataset['user'].unique()
    temp_2=[]
    
    for i in unique_values:
        temp = dataset[dataset['user'] == i]
        
        temp = temp.reset_index(drop=True)
        
        for i in range(len(temp)):
            try:
                past = temp['datetime'][i]
                future = temp['datetime'][i + 1]
                diff = future - past
                
                temp_2.append([temp['user'][i], temp['datetime'][i], temp['request'][i], temp['page_category'][i], diff])
                
            except:
                temp_2.append([temp['user'][i], temp['datetime'][i], temp['request'][i], temp['page_category'][i], timedelta(seconds=0)])
    
    dataset = pd.DataFrame(temp_2, columns=['user','datetime','request','page_category','time_on_page'])
    return dataset

weblog_df = time_on_page(weblog_df)

weblog_df['total_seconds'] = pd.to_timedelta(weblog_df['time_on_page']).view(np.int64) / 1e9

path = 'weblog_df.csv'
with open(path, 'w', encoding = 'utf-8-sig') as f:
  weblog_df.to_csv(f)

user_cluster = weblog_df.groupby('user')['request'].count()
user_cluster = user_cluster.reset_index()
user_cluster.columns = ['user', 'num_of_hits']

user_cluster2 = weblog_df.groupby('user')['total_seconds'].sum()
user_cluster2 = user_cluster2.reset_index()
user_cluster2.columns = ['user', 'total_seconds']

user_data = pd.merge(user_cluster, user_cluster2, on='user', how='inner')

path = 'kmeans_userdata.csv'

with open(path, 'w', encoding = 'utf-8-sig') as f:
  user_data.to_csv(f)

u_data = user_data[['num_of_hits','total_seconds']].values

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(u_data)

silhouette_score(u_data, kmeans.labels_)

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(u_data) for k in range(1,10)]

silhouette_scores = [silhouette_score(u_data, model.labels_)
                     for model in kmeans_per_k[1:]]

k = np.argmax(silhouette_scores) + 3

kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(u_data)

clusters = kmeans.fit_predict(u_data)

user_data["cluster"] = clusters

path = 'kmeans_results.csv'

with open(path, 'w', encoding = 'utf-8-sig') as f:
  user_data.to_csv(f)

weblog_path = os.path.join("weblog_df.csv")
weblog_df = pd.read_csv(weblog_path)

weblog_df = weblog_df.drop(["Unnamed: 0"], axis=1)

weblog_df['datetime'] = pd.to_datetime(weblog_df['datetime'], format='%Y-%m-%d %H:%M:%S%z')

weblog_df['time_on_page'] = ''

#Calculate users' time on page 
def time_on_page(dataset):
    unique_values = dataset['user'].unique()
    temp_2=[]
    
    for i in unique_values:
        temp = dataset[dataset['user'] == i]
        
        temp = temp.reset_index(drop=True)
        
        for i in range(len(temp)):
            try:
                past = temp['datetime'][i]
                future = temp['datetime'][i + 1]
                diff = future - past
                
                temp_2.append([temp['user'][i], temp['datetime'][i], temp['request'][i], temp['page_category'][i], diff])
                
            except:
                temp_2.append([temp['user'][i], temp['datetime'][i], temp['request'][i], temp['page_category'][i], timedelta(seconds=0)])
    
    dataset = pd.DataFrame(temp_2, columns=['user','datetime','request','page_category','time_on_page'])
    return dataset
  
weblog_df = time_on_page(weblog_df)

weblog_df['total_seconds'] = pd.to_timedelta(weblog_df['time_on_page']).view(np.int64) / 1e9

def is_idle(dataset):
    is_idle = []

    for i in range(0, len(dataset)):
        if (dataset["total_seconds"][i] >= (30 * 60)) or (
                dataset["total_seconds"][i] == 0 or dataset["page_category"][i] == "logout"):
            is_idle.append(True)
        else:
            is_idle.append(False)

    dataset["is_idle"] = is_idle
    return dataset

weblog_df = is_idle(weblog_df)

def session_identification(dataset):
    session_identification = []
    count = 1
    for i in range(len(dataset)):
        if (dataset['is_idle'][i] == False):
                session_identification.append(count)
        else:
                session_identification.append(count)
                count += 1
    dataset['session_identification'] = session_identification
    return dataset

weblog_df = session_identification(weblog_df)

notactive = weblog_df[weblog_df.session_identification == 1].shape[0]
active = weblog_df[weblog_df.session_identification != 1].shape[0]

path = 'weblog_df_session.csv'

with open(path, 'w', encoding = 'utf-8-sig') as f:
  weblog_df.to_csv(f)

def create_path(dataset):
  df_paths = dataset.groupby(['user','session_identification'])['page_category'].aggregate(lambda x: x.tolist()).reset_index()

  df_last_interaction = dataset.drop_duplicates('session_identification', keep='last')[['session_identification']]
  df_paths = pd.merge(df_paths, df_last_interaction, how='left', on='session_identification')

  df_paths=df_paths.drop("session_identification", axis=1)
  df_paths=df_paths.rename(columns={"page_category":"path"})

  return df_paths

user_path = create_path(weblog_df)

user_path = user_path.dropna()

path = 'user_path.csv'

with open(path, 'w', encoding = 'utf-8-sig') as f:
  user_path.to_csv(f)

path_list = user_path['path'].tolist()

a = TransactionEncoder()
a_data = a.fit(path_list).transform(path_list)
path_df = pd.DataFrame(a_data,columns=a.columns_)
path_df = path_df.replace(True,1)
path_df = path_df.replace(False,0)

path_df_apr = apriori(path_df, min_support = 0.04, use_colnames = True)

asc_df = association_rules(path_df_apr, metric = "confidence", min_threshold = 0.01)

asc_df = asc_df.drop(["antecedent support", "consequent support", "leverage", "conviction"], axis=1)

path = 'asc_support.csv'

with open(path, 'w', encoding = 'utf-8-sig') as f:
  path_df_apr.to_csv(f)

path = 'asc_confidence.csv'

with open(path, 'w', encoding = 'utf-8-sig') as f:
  asc_df.to_csv(f)


#Objective 3: Predict User Behaviours
df_path = os.path.join("kmeans_results.csv")
df = pd.read_csv(df_path)

df = df.drop(['Unnamed: 0'], axis=1)

X = df.drop(['cluster', 'user'], axis=1)

y = df['cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)

y_pred_train_gini = clf_gini.predict(X_train)

@app.route("/")
# This fuction for rendering the table
def index():
    #Statistics
    stats = {
        'num_of_users': weblog_df['user'].nunique(),
        'num_of_pages': weblog_df['page_category'].nunique(),
        'rev_table': asc_df. \
            head(10).reset_index().to_html(
            classes=['table thead-light table-striped table-bordered table-hover table-sm'])
    }

    #Bar Plot - Most Popular Date
    plt.rcParams['figure.figsize'] = (10, 6)
    color = plt.cm.winter(np.linspace(0, 1, 50))
    weblog['user'].value_counts().head(10).plot.barh(color = color)
    plt.title('Top 10 Active Users IP Address', fontsize = 20)
    plt.xlabel('Frequency of Users Visiting the Website', fontsize = 12)
    plt.ylabel('Users IP Address', fontsize = 12)
    plt.savefig('active_user.png',bbox_inches="tight") 

    # bagian ini digunakan untuk mengconvert matplotlib png ke base64 agar dapat ditampilkan ke template html
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    plt.close()
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    result = str(figdata_png)[2:-1]

    # Line Chart - Page Frequency
    plt.figure(figsize=(10, 6))
    page = page_cluster['page_category']
    hits = page_cluster['num_of_hits']

    pages = plt.plot(page, hits, color='red', marker='o')
    plt.title('The Frequency of Pages Visited', fontsize=20)
    # pages.set_xticks(page)
    # pages.set_xticklabels(hits, rotation=45)
    plt.xlabel('Page Categories', fontsize=12)
    plt.ylabel('Total Number of Each Pages Visited', fontsize=12)
    plt.show()
    plt.savefig('pages_visited.png', bbox_inches="tight")

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    plt.close()
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    result2 = str(figdata_png)[2:-1]

    # Pie Chart - User Satisfaction Level
    plt.rcParams['figure.figsize'] = (10, 6)
    page_data = [active, notactive]
    page_label = ['Satisfied Users', 'Unsatisfied Users']

    explode = (0, 0.1)
    plt.pie(page_data, explode=explode, labels=page_label, autopct='%1.1f%%')
    plt.title('Users Satisfaction Level', fontsize=20)
    plt.axis('equal')
    plt.legend(['Satisfied Users', 'Unsatisfied Users'], loc="upper right")
    plt.savefig('satisfaction.png', bbox_inches="tight")

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    plt.close()
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    result3 = str(figdata_png)[2:-1]

    # Scatter Plot - K-means Clustering
    plt.figure(figsize=(10, 6))
    sns.scatterplot(u_data[y_kmeans == 0, 0], u_data[y_kmeans == 0, 1], color='yellow', label='Cluster 0', s=50)
    sns.scatterplot(u_data[y_kmeans == 1, 0], u_data[y_kmeans == 1, 1], color='blue', label='Cluster 1', s=50)
    sns.scatterplot(u_data[y_kmeans == 2, 0], u_data[y_kmeans == 2, 1], color='green', label='Cluster 2', s=50)
    sns.scatterplot(u_data[y_kmeans == 3, 0], u_data[y_kmeans == 3, 1], color='orange', label='Cluster 3', s=50)
    sns.scatterplot(u_data[y_kmeans == 4, 0], u_data[y_kmeans == 4, 1], color='cyan', label='Cluster 4', s=50)
    sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red',
                    label='Centroids', s=300, marker='^')
    plt.grid(False)
    plt.title('Clusters of User Groups', fontsize=20)
    plt.xlabel('Number of Hits', fontsize=12)
    plt.ylabel('Total Seconds', fontsize=12)
    plt.legend()
    plt.savefig('kmeans_cluster.png', bbox_inches="tight")

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    plt.close()
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    result4 = str(figdata_png)[2:-1]

    # Pie Chart - K-means Clustering
    plt.figure(figsize=(10, 6))
    user_data['cluster'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Results of K-Means Clustering', fontsize=20)
    plt.legend(['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'], loc="upper right")
    plt.savefig('kmeans_count.png', bbox_inches="tight")

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    plt.close()
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    result5 = str(figdata_png)[2:-1]

    # Decision Tree
    #dot_data = tree.export_graphviz(clf_gini, out_file=None,
    #                                feature_names=X_train.columns,
    #                                filled=True, rounded=True,
    #                                special_characters=True)
    #graphviz.Source(dot_data, format="png")
    # Visualize decision-trees
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf_gini.fit(X_train, y_train))
    plt.savefig('decision_graph.png', bbox_inches="tight")

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    plt.close()
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    result6 = str(figdata_png)[2:-1]

    # Tambahkan hasil result plot pada fungsi render_template()
    return render_template('index.html', stats=stats, result=result, result2=result2, result3=result3, result4=result4, result5=result5, result6=result6)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

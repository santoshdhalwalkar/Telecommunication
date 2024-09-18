import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
# Display the DataFrame in Streamlit with container to adjust the width

#Display header
st.markdown("<header> Tellco Data Analystics</header>", unsafe_allow_html=True)

satisfaction_dashboard = pd.read_csv(r"C:\Users\santosh.dhalwalkar\Desktop\internproject\tellco\my_streamlit_app\data\engagement_scores.csv")
experience_dashboard = pd.read_csv(r"C:\Users\santosh.dhalwalkar\Desktop\internproject\tellco\my_streamlit_app\data\experience_Score.csv")
engagement_dashboard = pd.read_csv(r"C:\Users\santosh.dhalwalkar\Desktop\internproject\tellco\my_streamlit_app\data\Satisfaction_Score.csv")
telecome_data = pd.read_csv(r"C:\Users\santosh.dhalwalkar\Desktop\internproject\tellco\my_streamlit_app\data\telecomdata.csv")



# Row A
a1, a2, a3, a4 = st.columns(4)


with a1:
    st.image(Image.open(r'C:\Users\santosh.dhalwalkar\Desktop\internproject\tellco\my_streamlit_app\images\logo.png'))

with a2:
    st.write("#### Top 10 most engaged users per application")

    telecome_data['Youtube_Total'] = telecome_data['YoutubeDL_Bytes'] + telecome_data['YoutubeUL_Bytes']
    telecome_data['Netflix_Total'] = telecome_data['NetflixDL_Bytes'] + telecome_data['NetflixUL_Bytes']
    telecome_data['Gaming_Total'] = telecome_data['GamingDL_Bytes'] + telecome_data['GamingUL_Bytes']
    telecome_data['Other_Total'] = telecome_data['OtherDL_Bytes'] + telecome_data['OtherUL_Bytes']


    
    applications = ['Gaming_Total',
                    'Youtube_Total',
                    'Netflix_Total', 'Other_Total', ]
    app_top_users = []
    for app in applications:
        top_users = telecome_data.groupby('MSISDNNumber')[app].sum().nlargest(10)
        temp_df = pd.DataFrame({
            'MSISDNNumber': top_users.index,
            'Usage': top_users.values,
            'Application': app
        })
    app_top_users.append(temp_df)
    final_df = pd.concat(app_top_users)
    with st.container(height= 250, border=0 ):        
         st.dataframe(final_df)  # You can modify the width as needed


with a3:
    st.write("#### Top 10 most engaged users per application" )
    applications = ['Gaming_Total',
                    'Youtube_Total',
                    'Netflix_Total', 'Other_Total',]
    total_traffic = telecome_data[applications].sum()
    top_3_apps = total_traffic.nlargest(3)
    top_3_apps_names = top_3_apps.index.tolist()
    top_3_apps_df = pd.DataFrame({
        'Application': top_3_apps_names,
        'Total Traffic': top_3_apps.values
    })
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(top_3_apps, labels=top_3_apps.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', 3))
    ax.set_title('Top 3 Most Used Applications by Total Traffic')

    with st.container(height= 300, border=0 ):
    
      
         st.pyplot(fig)  # Pass the figure object to st.pyplot()
    plt.clf() # You can modify the width as needed

with a4:
    st.write("### Top 3 applications by total traffic:")
    applications = ['Gaming_Total', 'Youtube_Total', 'Netflix_Total', 'Other_Total']

    total_traffic = telecome_data[applications].sum()

    top_3_apps = total_traffic.nlargest(3)
    top_3_apps_names = top_3_apps.index.tolist()

    for app in top_3_apps_names:
        st.write(f"{app}: {top_3_apps[app]}")

    # Plot a bar chart for the top 3 applications
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_3_apps.index, y=top_3_apps.values, palette='viridis')
    plt.title('Top 3 Most Used Applications by Total Traffic')
    plt.xlabel('Application')
    plt.ylabel('Total Traffic (Bytes)')

    # Display the plot in Streamlit
    st.pyplot(plt)

# Row B
b1,b2,b3, = st.columns(3)

with b1:
    st.write("#### The average throughput per handset type")
      # Aggregate data per customer
    df_aggregated = telecome_data.groupby('MSISDNNumber').agg({
            'TCPDLRetransVol_Bytes': 'mean',
            'TCPULRetransVol_Bytes': 'mean',
            'AvgRTT_DL': 'mean',
            'AvgRTT_UL': 'mean',
            'AvgBearerTP_DL': 'mean',
            'AvgBearerTP_UL': 'mean',
            'BearerId': 'first'  # Assuming BearerId corresponds to the handset type
        }).reset_index()
    df_aggregated['Avg_TCP_Retransmission'] = (df_aggregated['TCPDLRetransVol_Bytes'] + df_aggregated['TCPULRetransVol_Bytes']) / 2
    df_aggregated['Avg_RTT'] = (df_aggregated['AvgRTT_DL'] + df_aggregated['AvgRTT_UL']) / 2
    df_aggregated['Avg_Throughput'] = (df_aggregated['AvgBearerTP_DL'] + df_aggregated['AvgBearerTP_UL']) / 2
    df_result = df_aggregated[['MSISDNNumber', 'BearerId', 'Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']]
    throughput_distribution = df_result.groupby('BearerId')['Avg_Throughput'].mean()

        
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 8))
    throughput_distribution.plot(kind='bar', ax=ax)
    ax.set_title('Average Throughput per Handset Type')
    ax.set_xlabel('Handset Type (BearerId)')
    ax.set_ylabel('Average Throughput')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    with st.container( border=0 ):
         st.pyplot(fig) 

with b2:
    st.write("### The average TCP retransmission view per handset type")
    tcp_retransmission_per_handset = df_result.groupby('BearerId')['Avg_TCP_Retransmission'].mean()
    tcp_retransmission_per_handset_sorted = tcp_retransmission_per_handset.sort_values()

    threshold = np.median(tcp_retransmission_per_handset_sorted)
    colors = ['green' if value <= threshold else 'red' for value in tcp_retransmission_per_handset_sorted]
    fig, ax = plt.subplots(figsize=(10, 8))
    tcp_retransmission_per_handset_sorted.plot(kind='bar', color=colors, ax=ax)

    ax.set_title('Average TCP Retransmission per Handset Type (Sorted)')
    ax.set_xlabel('Handset Type (BearerId)')
    ax.set_ylabel('Average TCP Retransmission')

    ax.set_xticklabels(tcp_retransmission_per_handset_sorted.index, rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.axhline(threshold, color='blue', linewidth=1, linestyle='--', label='Median Threshold')
    ax.legend(['Median Threshold', 'Good (Low Retransmission)', 'Bad (High Retransmission)'], loc='upper right')
    plt.tight_layout()
    with st.container( border=0 ):
         st.pyplot(fig) 

with b3:
    st.write("###  k-means clustering")

    df_clustering = df_aggregated[['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_aggregated['Cluster'] = kmeans.fit_predict(df_clustering)
    cluster_centers = kmeans.cluster_centers_
    cluster_counts = df_aggregated['Cluster'].value_counts()


        # Set the style for seaborn
    sns.set(style="whitegrid")

    # Create a 3D scatter plot to visualize the clusters
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Define the colors for each cluster
    colors = ['red', 'green', 'blue']

    # Plot each cluster
    for cluster in range(3):
        cluster_data = df_aggregated[df_aggregated['Cluster'] == cluster]
        ax.scatter(cluster_data['Avg_TCP_Retransmission'],
                cluster_data['Avg_RTT'],
                cluster_data['Avg_Throughput'],
                c=colors[cluster],
                label=f'Cluster {cluster}',
                s=50)

    # Set labels and title
    ax.set_xlabel('Avg TCP Retransmission (Bytes)')
    ax.set_ylabel('Avg RTT (ms)')
    ax.set_zlabel('Avg Throughput (Mbps)')
    ax.set_title('3D Scatter Plot of User Experience Clusters')

    # Add a legend
    ax.legend()

    plt.tight_layout()
    with st.container( border=0 ):
         st.pyplot(fig)


# Row C

c1,c2,c3 = st.columns(3)

with c1:
    st.write("### Correlation heatmap for the engagement, experience, and satisfaction scores.")
    df_scores = telecome_data[['engagement_score', 'Experience_Score', 'Satisfaction_Score']]

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_scores.corr(), annot=True, cmap="coolwarm", ax=ax_heatmap)
    ax_heatmap.set_title('Correlation Heatmap of Engagement, Experience, and Satisfaction')

    plt.tight_layout()
    with st.container( border=0 ):
         st.pyplot(fig_heatmap)
         

with c2:
    st.write("### Feature importance plot")

    X = telecome_data[['engagement_score', 'Experience_Score']]  # Features
    y = telecome_data['Satisfaction_Score']  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.write(f"R-squared: {r2}")

    importance = pd.Series(model.feature_importances_, index=X.columns)
    fig_importance, ax_importance = plt.subplots()
    importance.plot(kind='barh', ax=ax_importance, color='skyblue')
    ax_importance.set_title('Feature Importance')
    ax_importance.set_xlabel('Importance Score')

    plt.tight_layout()
    with st.container( border=0 ):
         st.pyplot(fig_importance)

with c3:

    st.write("### Predicted vs Actual Satisfaction Scores")
    fig_pred_actual, ax_pred_actual = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax_pred_actual, color='blue')
    ax_pred_actual.set_title('Predicted vs Actual Satisfaction Scores')
    ax_pred_actual.set_xlabel('Actual Satisfaction Score')
    ax_pred_actual.set_ylabel('Predicted Satisfaction Score')

    ax_pred_actual.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

    plt.tight_layout()
    with st.container( border=0 ):
         st.pyplot(fig_pred_actual)  
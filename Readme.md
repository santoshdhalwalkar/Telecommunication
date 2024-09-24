Task 1 - User Overview Analysis
●	Aggregate per user information number of xDR sessions, Session duration, the total download (DL) and 	the total data volume  upload (UL) data 

Conduct exploratory data analysis

○	Describe all relevant variables and associated data types
○	Analyze the basic metrics (mean, median, etc) in the Dataset 
○	Conduct a Non-Graphical Univariate Analysis 
○	Conduct a Graphical Univariate Analysis 
○	Conduct a Graphical Univariate Analysis 
○	Variable transformations
○	Correlation Analysis 
○	Dimensionality Reduction – perform a principal component analysis to reduce the dimensions

Task 2 - User Engagement Analysis
Track the user’s engagement metrics
●	sessions frequency 
●	the duration of the session 
●	The session total traffic (download and upload (bytes))
●	Aggregate the above metrics per customer ID (MSISDN)
●	Top 10 customers per engagement metric 
●	Run a k-means (k=3) to classify customers into three groups of engagement. 
●	Compute the minimum, maximum, average & total non-normalized metrics
●	Aggregate user total traffic per application and derive the top 10 most engaged users per application
●	lot the top 3 most used applications using appropriate charts.  
●	The optimized value of k useing the elbow method 

Task 3 - Experience Analytics
Expected to focus on network parameters like TCP retransmission, Round Trip Time (RTT), Throughput, and the customers’ device characteristics like the handset type to conduct a deep user experience analysis
●	Aggregate, per customer with above metrics
●	Compute & list 10 of the top, bottom, and most frequent:
●	The distribution of the average throughput per handset type
●	The average TCP retransmission view per handset type
●	Using the experience metrics above, perform a k-means clustering (where k = 3) to segment users into groups of experiences

Task 4 - Satisfaction Analysis

Expected  to analyze customer satisfaction in depth
●	The engagement score as the Euclidean distance between the user data point & the less engaged cluster
●	The experience score as the Euclidean distance between the user data point & the worst experience cluster
●	The average of both engagement & experience scores as  the satisfaction score & report the top 10 satisfied customer 
●	Build a regression model  to predict the satisfaction score of a customer
●	Run a k-means (k=2) on the engagement & the experience score. 
●	Aggregate the average satisfaction & experience score per cluster. 

1. You open the jpynb file with colab
2. To see dashboard I use streamlit in VS code  you can use that to create dashboard by creating .py file
3. Have dashboard images
4. ppt presention for project





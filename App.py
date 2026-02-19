
from flask import Flask, render_template,request,session
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

matplotlib.use('agg')
import numpy as np
import pickle



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/adminlogin')
def AdminLogin():
    return render_template('AdminApp/AdminLogin.html')

@app.route('/AdminAction', methods=['POST'])
def AdminAction():
    if request.method == 'POST':
        username=request.form['username']
        password=request.form['password']

        if username=='Admin' and password=='Admin':
            return render_template("AdminApp/AdminHome.html")
        else:
            context={'msg':'Login Failed..!!'}
            return render_template("AdminApp/AdminLogin.html",**context)

@app.route('/AdminHome')
def AdminHome():
    return render_template("AdminApp/AdminHome.html")

@app.route('/Upload')
def Upload():
    return render_template("AdminApp/Upload.html")



UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global data,filepath
@app.route('/UploadAction', methods=['POST'])
def UploadAction():
    global data,filepath
    if 'dataset' not in request.files:
        return "No file part"
    file = request.files['dataset']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    data = pd.read_csv(filepath)
    columns = data.columns.tolist()
    rows = data.head(20).values.tolist()
    return render_template('AdminApp/ViewDataset.html', columns=columns, rows=rows)

global data, X_train, X_test, y_train, y_test
@app.route('/preprocess')
def preprocess():
    global data, X_train, X_test, y_train, y_test,data,X,y

    data = pd.read_csv("Dataset/sentimentdataset.csv")
    data.dropna(inplace=True)

    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # Initialize the TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

    # Apply TF-IDF to the tweet column
    X_tfidf = tfidf.fit_transform(data['Text'].astype(str))
    X = data['Text']
    y = data['Sentiment']

    # Step 3: Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Step 4: TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('label_encoder_fast.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    return render_template('AdminApp/SplitStatus.html', total=len(X), train=len(X_train),test=len(X_test))

@app.route("/Sentiment")
def Sentiment():
    global data
    sentiment_count = data['Sentiment'].value_counts()
    labels = sentiment_count.index
    sizes  = sentiment_count.values
    colors = ['#FFA500', '#4169E1', '#778899'] 

    # Create pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Sentiment Analysis')
    plt.axis('equal') 
    plt.savefig("Static/pie.png")
    plt.close()
    return render_template('AdminApp/Graph.html', filename="pie.png")

@app.route("/PlotformSentiment")
def PlotformSentiment():
    global data
    data['Sentiment'] = data['Sentiment'].str.lower().str.strip()
    data['Platform'] = data['Platform'].str.title().str.strip()
    # Filter to valid sentiment and platform values
    valid_sentiments = ['positive', 'negative', 'neutral']
    valid_platforms = ['Twitter', 'Facebook', 'Instagram']
    data = data[data['Sentiment'].isin(valid_sentiments) & data['Platform'].isin(valid_platforms)]
    # Create a pivot table for plotting
    pivot = data.pivot_table(index='Platform', columns='Sentiment', aggfunc='size', fill_value=0)
    # Plot grouped bar chart
    pivot = pivot[valid_sentiments]  # Ensure correct order
    pivot.plot(kind='bar', figsize=(10, 6), color=['green', 'red', 'gray'])

    plt.title('Sentiment Analysis Across Platforms')
    plt.xlabel('Social Media Platform')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=0)
    plt.legend(title='Sentiment')
    plt.savefig("Static/Bar.png")
    plt.tight_layout()
    plt.close()
    return render_template('AdminApp/Graph.html', filename="Bar.png")


@app.route("/Likes")
def Likes():
    global data
    # Group by platform and sum likes/retweets
    grouped = data.groupby('Platform')[['Likes', 'Retweets']].sum().loc[['Twitter', 'Facebook', 'Instagram']]

    # Plot line graph
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped['Likes'], marker='o', label='Likes', color='blue')
    plt.plot(grouped.index, grouped['Retweets'], marker='o', label='Retweets', color='orange')

    plt.title('Likes and Retweets Across Platforms')
    plt.xlabel('Platform')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Static/LinePlot.png")
    plt.close()
    return render_template('AdminApp/Graph.html', filename="LinePlot.png")

global gbc_acc,rf_acc
@app.route('/gbc')
def gbc():
    global gbc_acc,rf_acc
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)
    pred = gb_model.predict(X_test)
    gbc_acc = accuracy_score(y_test, pred) * 100

    #RandomForest Algorithm
    rm = RandomForestClassifier()
    rm.fit(X_train, y_train)
    rf_pred = rm.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred) * 100
    return render_template('AdminApp/AlgorithmStatus.html', msg="Algorithm Model Generated Successfully..!!")




@app.route('/comparison')
def comparison():
    models = ['Gradient Boosting', 'Random Forest',]
    accuracies = [gbc_acc, rf_acc,]

    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)  # Adjust y-axis to match the accuracy range
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Static/model_accuracy.png')
    plt.close()
    return render_template('AdminApp/Graph.html',filename="model_accuracy.png")


if __name__ == '__main__':
    app.run(debug=True)



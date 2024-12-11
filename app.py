from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

# Load dataset
smartphones_data = pd.read_csv('smartphones.csv')
print(smartphones_data.head())

# Normalize the 'price' column
scaler = MinMaxScaler()
smartphones_data['price_normalized'] = scaler.fit_transform(smartphones_data[['price']].fillna(0))

# Function to recommend smartphones based on a given price
def recommend_smartphones(user_budget, top_n=5):
    # Normalize the user's budget
    normalized_budget = scaler.transform(pd.DataFrame([[user_budget]], columns=['price']))[0][0]
    
    # Calculate the distance between the user's budget and each smartphone's price
    smartphones_data['distance'] = smartphones_data['price_normalized'].apply(
        lambda x: euclidean_distances([[x]], [[normalized_budget]])[0][0]
    )
    
    # Sort by distance and get the top N recommendations
    recommendations = smartphones_data.sort_values('distance').head(top_n)
    
    # Replace NaN values in specific columns
    recommendations['avg_rating'] = recommendations['avg_rating'].fillna("Tidak tersedia")
    recommendations['fast_charging_available'] = recommendations['fast_charging_available'].fillna("Tidak tersedia")
    
    # Replace 1 with 'yes' and 0 with 'no' in the 'fast_charging_available' column
    recommendations['fast_charging_available'] = recommendations['fast_charging_available'].replace({1: 'yes', 0: 'no'})
    
    # Return only the required columns (remove 'brand_name')
    return recommendations[['model', 'processor_brand', 'fast_charging_available', 'price', 'avg_rating']].to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        try:
            user_budget = float(request.form['budget'])
            top_n = int(request.form['top_n'])
            recommendations = recommend_smartphones(user_budget, top_n)
        except ValueError:
            recommendations = []
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, port=8000)

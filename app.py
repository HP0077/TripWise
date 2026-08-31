# ==============================================================================
# TripWise - Flask Web Application Backend
# Handles user preference input, applies preprocessing, runs the ML ensemble,
# and returns personalized destination recommendations.
# ==============================================================================

import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import pycountry_convert as pc

app = Flask(__name__)

# Step 1: Load pre-trained models, encoders, and destination names
print("Loading trained machine learning models...")
model = joblib.load('model.pkl')
mlb_encoders = joblib.load('mlb_encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load destination mapping from destination_ids.txt
destinations = {}
with open('destination_ids.txt', 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        destinations[idx] = line.strip()


def get_continent_code(destination_full_name):
    """
    Helper function to map a destination name ('City, Country') to its continent code:
    'a': Asia, 'e': Europe, 'n': North America, 's': South America, 'f': Africa, 'o': Oceania
    """
    parts = destination_full_name.split(',')
    country = parts[-1].strip()

    # Manual overrides for specific naming conventions
    if country in ['United States', 'USA']:
        return 'n'
    if country in ['United Kingdom', 'UK', 'Scotland', 'England']:
        return 'e'
    if country in ['South Korea', 'Korea']:
        return 'a'
    if country in ['Czech Republic']:
        return 'e'

    try:
        alpha2 = pc.country_name_to_country_alpha2(country, cn_name_format="default")
        continent = pc.country_alpha2_to_continent_code(alpha2)
        continent_map = {
            'AS': 'a',
            'EU': 'e',
            'NA': 'n',
            'SA': 's',
            'AF': 'f',
            'OC': 'o'
        }
        return continent_map.get(continent, 'e')
    except Exception:
        # Default fallback to Europe if country lookup fails
        return 'e'


@app.route('/')
def index():
    """Serves the main frontend page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives user travel preferences via JSON POST request,
    formats them into model input features, predicts match probabilities,
    and returns the top recommended destinations.
    """
    data = request.json or {}

    # Extract single-choice inputs
    user_budget = int(data.get('budget', -1))
    user_season = int(data.get('season', -1))
    user_activity = int(data.get('activity', -1))
    user_safety = int(data.get('safety', -1))
    user_popularity = int(data.get('popularity', -1))

    # Extract multi-choice inputs as string lists
    user_experiences = [str(x) for x in data.get('experiences', [])]
    user_scenery = [str(x) for x in data.get('scenery', [])]
    user_regions = [str(x) for x in data.get('region', [])]

    # Create DataFrame for single-choice features
    single_features_df = pd.DataFrame([{
        'form_b': user_budget,
        'form_c': user_season,
        'form_h': user_activity,
        'form_i': user_safety,
        'form_j': user_popularity
    }])

    # Multi-label binarize multi-choice features using pre-fitted encoders
    multi_features_dfs = []
    for col_name, values in [('form_f', user_experiences), ('form_g', user_scenery), ('form_rr', user_regions)]:
        encoder = mlb_encoders[col_name]
        binarized = encoder.transform([values])
        col_names = [f"{col_name}_{cls_name}" for cls_name in encoder.classes_]
        multi_features_dfs.append(pd.DataFrame(binarized, columns=col_names))

    # Combine all input features into single vector
    input_vector = pd.concat([single_features_df] + multi_features_dfs, axis=1)

    # Ensure all feature columns match training structure exactly
    for col in feature_columns:
        if col not in input_vector.columns:
            input_vector[col] = 0

    input_vector = input_vector[feature_columns]

    # Run soft-voting prediction to get class probabilities
    probabilities = model.predict_proba(input_vector)[0]
    class_dest_ids = model.classes_

    # Collect destination candidates and filter by region if requested
    candidate_destinations = []
    for idx, dest_id in enumerate(class_dest_ids):
        dest_name = destinations.get(dest_id, f"Destination {dest_id}")
        region_code = get_continent_code(dest_name)

        # Apply strict region filter if user selected specific regions
        if len(user_regions) > 0 and region_code not in user_regions:
            continue

        raw_prob = float(probabilities[idx])
        candidate_destinations.append({
            "name": dest_name,
            "match": raw_prob
        })

    # Sort destinations in descending order of match score
    candidate_destinations.sort(key=lambda item: item["match"], reverse=True)

    # Select top 6 recommended destinations
    top_recommendations = candidate_destinations[:6]

    # Normalize match percentage so top score is 95% and rest scale proportionately
    if top_recommendations:
        top_prob = top_recommendations[0]['match']
        if top_prob > 0:
            for item in top_recommendations:
                scaled_pct = round((item['match'] / top_prob) * 95.0, 1)
                item['match'] = max(scaled_pct, 12.0)

    # Build human-readable summary of the user's selected preferences
    budget_labels = {0: "$0-$49", 1: "$50-$99", 2: "$100-$249", 3: "$300+"}
    season_labels = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
    experience_labels = {
        '0': 'Beach', '1': 'Adventure', '2': 'Nature', '3': 'Culture',
        '4': 'Nightlife', '5': 'History', '6': 'Shopping', '7': 'Cuisine'
    }
    scenery_labels = {
        '0': 'Urban', '1': 'Rural', '2': 'Sea', '3': 'Mountain',
        '4': 'Lake', '5': 'Desert', '6': 'Plains', '7': 'Jungle'
    }
    region_labels = {
        'e': 'Europe', 'n': 'N. America', 'a': 'Asia',
        's': 'S. America', 'f': 'Africa', 'o': 'Oceania'
    }

    summary_parts = []
    if user_season in season_labels:
        summary_parts.append(season_labels[user_season])
    if user_budget in budget_labels:
        summary_parts.append(f"{budget_labels[user_budget]} budget")
    
    selected_exp = [experience_labels[k] for k in user_experiences if k in experience_labels]
    if selected_exp:
        summary_parts.append(", ".join(selected_exp))

    selected_scene = [scenery_labels[k] for k in user_scenery if k in scenery_labels]
    if selected_scene:
        summary_parts.append(f"{', '.join(selected_scene)} scenery")

    selected_reg = [region_labels[k] for k in user_regions if k in region_labels]
    if selected_reg:
        summary_parts.append(f"in {', '.join(selected_reg)}")

    if summary_parts:
        summary_message = "Top personalized matches for: " + " | ".join(summary_parts)
    else:
        summary_message = "Top personalized matches for your vacation:"

    return jsonify({
        "success": True,
        "destinations": top_recommendations,
        "summary": summary_message
    })


if __name__ == '__main__':
    # Run development server locally on port 5000
    app.run(debug=True, port=5000)

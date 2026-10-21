# ==============================================================================
# TripWise - Flask Web Application Backend
# Handles user preference input, applies preprocessing, runs the ML ensemble,
# and returns personalized destination recommendations.
# ==============================================================================

import os
import json
import warnings
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# Static files live in public/static: Vercel serves public/ from its CDN,
# and Flask serves the same folder at /static when running locally.
app = Flask(__name__, static_folder='public/static', static_url_path='/static')

# The models were fitted on a pandas DataFrame; at runtime they get a plain
# NumPy row in the same column order, so the feature-name check is not needed.
warnings.filterwarnings('ignore', message='X does not have valid feature names')

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

# Continent code for each destination the model can recommend, written by train_model.py
# ('a': Asia, 'e': Europe, 'n': North America, 's': South America, 'f': Africa, 'o': Oceania)
with open('destination_continents.json', 'r', encoding='utf-8') as file:
    continent_by_name = {destinations[int(dest_id)]: code for dest_id, code in json.load(file).items()}

# Held-out evaluation results produced by evaluate.py (shown in the "How it works" section)
model_metrics = None
if os.path.exists('metrics.json'):
    with open('metrics.json', 'r', encoding='utf-8') as file:
        model_metrics = json.load(file)


def get_continent_code(destination_full_name):
    """Returns the continent code for a destination name ('City, Country')."""
    return continent_by_name.get(destination_full_name)


# Valid answer codes for each questionnaire field (see Stravl dataset documentation)
SINGLE_CHOICE_OPTIONS = {
    'budget': range(4), 'season': range(4), 'activity': range(3),
    'safety': range(3), 'popularity': range(3)
}
MULTI_CHOICE_OPTIONS = {
    'experiences': {str(i) for i in range(8)},
    'scenery': {str(i) for i in range(8)},
    'region': {'a', 'e', 'n', 's', 'f', 'o'}
}


def parse_single_choice(data, field):
    """Returns the integer answer for a single-choice field, or -1 when unanswered."""
    value = data.get(field)
    if value is None or value == '':
        return -1
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be an integer")
    if value not in SINGLE_CHOICE_OPTIONS[field]:
        raise ValueError(f"'{field}' is out of range")
    return value


def parse_multi_choice(data, field):
    """Returns the list of answer codes for a multi-choice field."""
    values = data.get(field, [])
    if not isinstance(values, list):
        raise ValueError(f"'{field}' must be a list")
    values = [str(x) for x in values]
    if not set(values) <= MULTI_CHOICE_OPTIONS[field]:
        raise ValueError(f"'{field}' contains unknown options")
    return values


def build_input_vector(budget, season, activity, safety, popularity, experiences, scenery, regions):
    """Turns questionnaire answers into the 29-column feature vector the model was trained on."""
    row = {
        'form_b': budget,
        'form_c': season,
        'form_h': activity,
        'form_i': safety,
        'form_j': popularity
    }

    # Multi-label binarize multi-choice features using pre-fitted encoders
    for col_name, values in [('form_f', experiences), ('form_g', scenery), ('form_rr', regions)]:
        encoder = mlb_encoders[col_name]
        binarized = encoder.transform([values])[0]
        for cls_name, flag in zip(encoder.classes_, binarized):
            row[f"{col_name}_{cls_name}"] = int(flag)

    # One row, in exactly the column order used for training
    return np.array([[row.get(col, 0) for col in feature_columns]])


def build_example_vote():
    """
    Runs one example questionnaire through both halves of the ensemble so the
    "How it works" page can show real probabilities being averaged.
    """
    vector = build_input_vector(2, 2, 1, 1, 1, ['0', '7'], ['2'], [])
    ensemble_proba = model.predict_proba(vector)[0]
    member_proba = {name: est.predict_proba(vector)[0] for name, est in model.named_estimators_.items()}
    top = ensemble_proba.argsort()[::-1][:3]
    return [{
        "name": destinations[model.classes_[i]],
        "rf": round(float(member_proba['rf'][i]) * 100, 2),
        "et": round(float(member_proba['et'][i]) * 100, 2),
        "vote": round(float(ensemble_proba[i]) * 100, 2)
    } for i in top]


# Facts shown on the "How it works" page, computed from the loaded artifacts
CONTINENT_NAMES = {'e': 'Europe', 'a': 'Asia', 'n': 'North America', 's': 'South America', 'f': 'Africa', 'o': 'Oceania'}
continent_counts = {name: 0 for name in CONTINENT_NAMES.values()}
for dest_id in model.classes_:
    continent_counts[CONTINENT_NAMES[get_continent_code(destinations[dest_id])]] += 1

page_facts = {
    "survey_destinations": len(destinations),
    "model_destinations": len(model.classes_),
    "continent_counts": continent_counts,
    "feature_columns": feature_columns,
    "example_vote": build_example_vote(),
    "model_size_mb": round(os.path.getsize('model.pkl') / (1024 * 1024), 1),
}


@app.route('/')
def index():
    """Serves the main frontend page."""
    return render_template('index.html', metrics=model_metrics)


@app.route('/how-it-works')
def how_it_works():
    """Explains the data, features, models and evaluation behind the recommendations."""
    return render_template('how-it-works.html', metrics=model_metrics, facts=page_facts)


@app.errorhandler(404)
def page_not_found(error):
    """Serves the custom 404 page."""
    return render_template('404.html'), 404


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives user travel preferences via JSON POST request,
    formats them into model input features, predicts match probabilities,
    and returns the top recommended destinations.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Request body must be a JSON object"}), 400

    try:
        # Extract single-choice inputs
        user_budget = parse_single_choice(data, 'budget')
        user_season = parse_single_choice(data, 'season')
        user_activity = parse_single_choice(data, 'activity')
        user_safety = parse_single_choice(data, 'safety')
        user_popularity = parse_single_choice(data, 'popularity')

        # Extract multi-choice inputs as string lists
        user_experiences = parse_multi_choice(data, 'experiences')
        user_scenery = parse_multi_choice(data, 'scenery')
        user_regions = parse_multi_choice(data, 'region')
    except ValueError as err:
        return jsonify({"success": False, "error": str(err)}), 400

    input_vector = build_input_vector(
        user_budget, user_season, user_activity, user_safety, user_popularity,
        user_experiences, user_scenery, user_regions
    )

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

        candidate_destinations.append({
            "name": dest_name,
            "probability": float(probabilities[idx])
        })

    # Sort destinations in descending order of predicted probability
    candidate_destinations.sort(key=lambda item: item["probability"], reverse=True)

    # Select top 6 recommended destinations
    top_recommendations = candidate_destinations[:6]

    # Relative fit score: the top pick is 100, the rest scale proportionately
    if top_recommendations:
        top_prob = top_recommendations[0]['probability']
        for item in top_recommendations:
            item['match'] = round(item['probability'] / top_prob * 100.0, 1) if top_prob > 0 else 0.0
            item['probability'] = round(item['probability'], 5)

    # Build human-readable summary of the user's selected preferences
    budget_labels = {0: "$0-49", 1: "$50-99", 2: "$100-249", 3: "$300+"}
    season_labels = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
    popularity_labels = {0: "Off the beaten path", 1: "Classic spots", 2: "Mainstream & trendy"}
    experience_labels = {
        '0': 'Beach', '1': 'Adventure', '2': 'Nature', '3': 'Culture',
        '4': 'Nightlife', '5': 'History', '6': 'Shopping', '7': 'Food'
    }
    scenery_labels = {
        '0': 'City', '1': 'Countryside', '2': 'Sea', '3': 'Mountains',
        '4': 'Lakes', '5': 'Desert', '6': 'Plains', '7': 'Jungle'
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
    if user_popularity in popularity_labels:
        summary_parts.append(popularity_labels[user_popularity])
    
    selected_exp = [experience_labels[k] for k in user_experiences if k in experience_labels]
    if selected_exp:
        summary_parts.append(", ".join(selected_exp))

    selected_scene = [scenery_labels[k] for k in user_scenery if k in scenery_labels]
    if selected_scene:
        summary_parts.append(f"{', '.join(selected_scene)} views")

    selected_reg = [region_labels[k] for k in user_regions if k in region_labels]
    if selected_reg:
        summary_parts.append(f"in {', '.join(selected_reg)}")

    if summary_parts:
        summary_message = " · ".join(summary_parts)
    else:
        summary_message = "Any season · any budget · anywhere"

    return jsonify({
        "success": True,
        "destinations": top_recommendations,
        "summary": summary_message
    })


if __name__ == '__main__':
    # Run development server locally on port 5000
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', port=int(os.environ.get('PORT', 5000)))

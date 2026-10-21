# ==============================================================================
# TripWise - Machine Learning Travel Destination Recommender
# Model Training Script (Ensemble Learning: Random Forest + Extra Trees)
# Optimized for cloud deployment (Render, GitHub) with 225 global destinations
# ==============================================================================

import os
import ast
import json
import pandas as pd
import numpy as np
import joblib
import pycountry_convert as pc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier


def parse_string_list(text_value):
    """Safely converts string representation of lists into Python list of strings."""
    if pd.isna(text_value) or text_value == '[]':
        return []
    try:
        parsed = ast.literal_eval(text_value)
        return [str(item) for item in parsed]
    except Exception:
        return []


def parse_single_int(text_value):
    """Extracts single integer choice from list string representation, default -1 if missing."""
    if pd.isna(text_value) or text_value == '[]':
        return -1
    try:
        parsed = ast.literal_eval(text_value)
        if len(parsed) > 0:
            return int(parsed[0])
        return -1
    except Exception:
        return -1


def get_region_code(destination_full_name):
    """Maps destination country names to continent codes."""
    parts = destination_full_name.split(',')
    country = parts[-1].strip()

    if country in ['United States', 'USA']: return 'n'
    if country in ['United Kingdom', 'UK', 'Scotland', 'England']: return 'e'
    if country in ['South Korea', 'Korea']: return 'a'
    if country in ['Czech Republic']: return 'e'

    try:
        alpha2 = pc.country_name_to_country_alpha2(country, cn_name_format="default")
        continent = pc.country_alpha2_to_continent_code(alpha2)
        continent_map = {'AS': 'a', 'EU': 'e', 'NA': 'n', 'SA': 's', 'AF': 'f', 'OC': 'o'}
        return continent_map.get(continent, 'e')
    except Exception:
        return 'e'


def build_dataset(csv_path='Stravl_Travel_Preference_Data.csv'):
    """
    Loads the Stravl dataset and returns (X, y, user_ids, encoders, selected_destinations).
    Each row is one (user, liked destination) pair, so user_ids is kept for group-aware splitting.
    """
    print("Loading Stravl Travel Preference Dataset...")
    df = pd.read_csv(csv_path)

    with open('destination_ids.txt', 'r', encoding='utf-8') as f:
        all_dest_names = [line.strip() for line in f]

    # Step 1: Preprocess target column (yes_swipes)
    df['yes_swipes'] = df['yes_swipes'].apply(
        lambda x: [int(dest_id) for dest_id in ast.literal_eval(x)] if pd.notna(x) and x != '[]' else []
    )
    df_exploded = df.explode('yes_swipes').dropna(subset=['yes_swipes'])
    df_exploded['yes_swipes'] = df_exploded['yes_swipes'].astype(int)

    # Step 2: Select balanced destinations across all 6 continents
    counts = df_exploded['yes_swipes'].value_counts()
    dest_regions = {i: get_region_code(all_dest_names[i]) for i in range(len(all_dest_names)) if i < len(all_dest_names)}

    selected_destinations = []
    for r in ['a', 'e', 'n', 's', 'f', 'o']:
        d_list = [d for d in counts.index if d < len(all_dest_names) and dest_regions.get(d) == r and counts[d] >= 15]
        limit = 45 if r in ['a', 'e', 'n'] else 30
        selected_destinations.extend(d_list[:limit])

    print(f"Selected {len(selected_destinations)} balanced destinations across all continents.")
    data = df_exploded[df_exploded['yes_swipes'].isin(selected_destinations)].copy()

    # Step 3: Feature Preprocessing
    single_choice_features = ['form_b', 'form_c', 'form_h', 'form_i', 'form_j']
    for col in single_choice_features:
        data[col] = data[col].apply(parse_single_int)

    multi_choice_features = ['form_f', 'form_g', 'form_rr']
    for col in multi_choice_features:
        data[col] = data[col].apply(parse_string_list)

    encoders = {}
    encoded_feature_dfs = []

    for col in multi_choice_features:
        mlb = MultiLabelBinarizer()
        binary_matrix = mlb.fit_transform(data[col])
        col_names = [f"{col}_{cls_name}" for cls_name in mlb.classes_]
        encoders[col] = mlb
        encoded_df = pd.DataFrame(binary_matrix, columns=col_names, index=data.index)
        encoded_feature_dfs.append(encoded_df)

    X = pd.concat([data[single_choice_features]] + encoded_feature_dfs, axis=1)
    y = data['yes_swipes']

    print(f"Feature matrix shape: {X.shape}, Number of classes: {y.nunique()}")
    return X, y, data['id'], encoders, selected_destinations


def build_ensemble():
    """Balanced Random Forest + Extra Trees soft-voting ensemble used by the app."""
    rf_model = RandomForestClassifier(
        n_estimators=40,
        max_depth=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    et_model = ExtraTreesClassifier(
        n_estimators=40,
        max_depth=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    return VotingClassifier(
        estimators=[('rf', rf_model), ('et', et_model)],
        voting='soft'
    )


def main():
    X, y, _, encoders, selected_destinations = build_dataset()

    # Step 4: Train Ensemble Voting Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    print("Training Balanced Random Forest & Extra Trees Ensemble...")
    ensemble_model = build_ensemble()
    ensemble_model.fit(X_train, y_train)

    # Step 5: Save Compressed Model Artifacts for Render
    print("Saving compressed model artifacts (<50MB for cloud deployment)...")
    joblib.dump(ensemble_model, 'model.pkl', compress=3)
    joblib.dump(encoders, 'mlb_encoders.pkl')
    joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
    joblib.dump(selected_destinations, 'top_dests.pkl')

    # Continent per destination, so the web app can filter without pycountry at runtime
    with open('destination_ids.txt', 'r', encoding='utf-8') as f:
        all_dest_names = [line.strip() for line in f]
    continents = {str(int(d)): get_region_code(all_dest_names[d]) for d in selected_destinations}
    with open('destination_continents.json', 'w', encoding='utf-8') as f:
        json.dump(continents, f, indent=0, sort_keys=True)

    model_size_mb = os.path.getsize('model.pkl') / (1024 * 1024)
    print(f"Training complete! model.pkl size: {model_size_mb:.2f} MB (Ready for GitHub & Render).")


if __name__ == '__main__':
    main()

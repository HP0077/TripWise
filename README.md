# TripWise - Machine Learning Travel Destination Recommender

TripWise is an AI-powered travel recommendation web application built with Python, Flask, Scikit-Learn, and Vanilla HTML/CSS/JavaScript. It uses an ensemble of Random Forest and Extra Trees Voting Classifiers trained on real traveler preferences to provide personalized vacation matches.

---

## 🌟 Key Features

- **Trained ML Ensemble:** Random Forest + Extra Trees Voting Classifier with balanced class weights.
- **Global Coverage:** 225+ handpicked destinations spanning 6 continents (Asia, Europe, North America, South America, Africa, Oceania).
- **Zero GPU / Lightweight UI:** Fast 60fps animations with the "Sunset Wanderlust" theme.
- **Live Image Integration:** Dynamically fetches destination photos using the Wikipedia Search API.

---

## 💻 Running Locally

### 1. Prerequisites
Make sure Python 3.9+ is installed.

### 2. Install Dependencies
Open a terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the Model
If you ever want to re-run training from the dataset:
```bash
python train_model.py
```

### 4. Start the Web Server
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 🚀 Deploying to Render (render.com)

This repository is pre-configured with `Procfile`, `render.yaml`, and `requirements.txt` for one-click deployment on Render.

### Step 1: Push Project to GitHub
1. Initialize git in your project folder (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Deploy TripWise ML Recommender"
   ```
2. Create a new repository on [GitHub](https://github.com/new).
3. Push your code to GitHub:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Create Web Service on Render
1. Go to **[render.com](https://render.com/)** and log in (or sign up with GitHub).
2. Click **"New +"** in the top right and select **"Web Service"**.
3. Choose **"Build and deploy from a Git repository"** and select your GitHub repository.
4. Fill in the deployment details:
   - **Name:** `tripwise` (or any name you like)
   - **Language / Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** `Free`
5. Click **"Deploy Web Service"**.

Render will automatically install the requirements, start Gunicorn, and provide you with a live public URL (e.g. `https://tripwise.onrender.com`).

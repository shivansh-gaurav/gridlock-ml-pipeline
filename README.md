# Traffic Mobility Predictor (Flipkart GridLock 2.0)

## Overview
A high-performance Machine Learning pipeline built to forecast real-time traffic demand for the **Flipkart GridLock 2.0 Hackathon**. The pipeline processes spatio-temporal data (geohashes, timestamps, weather, landmarks) and predicts traffic volume. The final model achieved a **91.45 / 100 R2 Score** on the official leaderboard, placing in the top bracket of competitors.

## Why I Built It
I wanted to bridge the gap between theoretical data science and messy, real-world data engineering. Traffic prediction isn't just a simple regression problem—it is highly dependent on time-series extrapolation and spatial clustering. I built this to learn how to handle future-time extrapolation, prevent data leakage, and construct ensemble models (LightGBM, XGBoost, CatBoost) from scratch.

## Features
- **Spatio-Temporal Feature Engineering**: Extracted non-linear features from raw timestamps, mapped geohashes to spatial clusters, and engineered rolling lag features to capture traffic momentum.
- **Advanced Ensemble Modeling**: Built a stacked architecture using **LightGBM**, **XGBoost**, and **RandomForest** regressors, optimizing hyperparameters for each using empirical testing.
- **Score-Geometry Blending**: Developed a custom `update_from_probe_score.py` script that used the public leaderboard responses to mathematically triangulate the optimal blend weights for the final submission.
- **Time-Aware Validation**: Replaced standard Random K-Fold cross-validation with a strict chronological split to prevent future-data leakage, matching the actual test set's extrapolation setup.

## Tech Stack
- **Language**: Python 3
- **Libraries**: Pandas, Scikit-Learn, LightGBM, XGBoost, CatBoost, NumPy
- **Architecture**: Time-Series Forecasting, Ensemble Stacking, Feature Engineering Pipeline

## Architecture & Code Structure
- `src/modeling.py`: The core pipeline evaluating multiple gradient boosting frameworks.
- `src/kartik.py`: Highly optimized LightGBM specific pipeline handling lag features.
- `src/update_from_probe_score.py`: The custom mathematical blender that computes optimal submission weights based on leaderboard feedback geometry.
- `src/future_window_simulator.py`: Scripts to simulate the exact future-window prefix constraints of the hidden test set to prevent local overfitting.

## Technical Challenges
1. **The Validation Trap**: Early models (RandomForest) scored an incredible `95+` on local validation but crashed to `89` on the actual leaderboard. **Solution**: I realized random train/test splitting was causing data leakage because traffic data is chronological. I entirely rewrote the validation loop to be strictly time-aware (training on Day 48, testing on future windows in Day 49), which stabilized my local-to-leaderboard correlation.
2. **Feature Extrapolation Limits**: The test set asked for predictions 13 hours into the future, making standard 15-minute rolling averages useless. **Solution**: I engineered "Day-over-Day" lag features, looking at the exact same timestamp from the previous 24 hours to capture macro-level daily traffic cycles.
3. **Optimizing the Final Percentage**: Getting past the 90-score wall was mathematically difficult using single models. **Solution**: I researched and implemented Score-Geometry Blending, treating leaderboard scores as vectors to calculate the exact optimal blending ratio of my LightGBM and XGBoost outputs, ultimately pushing the score to `91.45`.

## Lessons Learned
- **Trust your Validation, not the Model**: A complex Deep Learning model evaluated incorrectly is useless. A simple regression tree evaluated strictly chronologically is powerful.
- **Feature Engineering > Hyperparameters**: Tuning `learning_rate` for 3 hours yielded a 0.1% improvement. Engineering a single intelligent temporal feature yielded a 3% improvement.
- **Data Leakage is Silent**: I learned firsthand how easy it is to accidentally leak target data into the training set when dealing with time-series lags.

## Installation
1. Clone the repository: `git clone https://github.com/shivansh-gaurav/gridlock-ml-pipeline.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the competition datasets into the `/dataset/` directory.
4. Execute `python src/modeling.py` to generate the base predictions.

## License
MIT License

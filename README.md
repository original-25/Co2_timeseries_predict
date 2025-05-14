
# CO2 Prediction with Linear Regression

## 🚀 Overview
This project demonstrates the power of machine learning to predict CO₂ concentration levels based on historical data using a Linear Regression model. The code leverages the CO2 emissions dataset (`co2.csv`), processes the time series data, and trains a model to forecast future CO₂ concentrations.

## 📈 Key Features
- **Data Preprocessing**: Clean and interpolate missing values.
- **Feature Engineering**: Create lag features (time-shifted columns) for better prediction accuracy.
- **Modeling**: Use Linear Regression to predict CO₂ levels.
- **Evaluation**: Assess model performance with MAE, MSE, and R² score.
- **Prediction Loop**: Forecast CO₂ levels iteratively to simulate real-time predictions.

## ⚙️ Requirements
- Python 3.x
- `pandas` for data manipulation
- `matplotlib` for plotting
- `scikit-learn` for machine learning tasks

To install the necessary packages, run:

```bash
pip install pandas matplotlib scikit-learn
```

## 📊 Data Description
This project uses the `co2.csv` file containing CO₂ concentration data over time. The `time` column represents the timestamp, and the `co2` column contains the CO₂ concentration values.

## 🛠️ Workflow

### Load and Process Data:
- The `co2.csv` file is loaded and the `time` column is converted to datetime format.
- Missing values in the CO₂ concentration data are interpolated to ensure a smooth time series.

### Feature Engineering:
- Lag features are created, where the past CO₂ values are shifted over time (with a default window size of 5). This helps in predicting future CO₂ concentrations.
- The target variable (`target`) is the CO₂ concentration shifted by the window size.

### Train/Test Split:
- The data is split into training and testing sets, with an 80%/20% ratio.

### Linear Regression Model:
- A linear regression model is trained on the training data (`x_train`, `y_train`).
- The model is evaluated on the testing set (`x_test`, `y_test`) using the following metrics:
  - **MAE (Mean Absolute Error)**
  - **MSE (Mean Squared Error)**
  - **R² score (Coefficient of Determination)**

### Iterative Prediction:
- The trained model is used to predict CO₂ levels for a sample input iteratively, simulating real-time predictions based on previous outputs.

## 🏃‍♂️ Running the Code
Once all the dependencies are installed, you can simply run the Python script to see the output:

```bash
python model_test.py
```

### Example Output:

```
MAE: 5.124
MSE: 34.98
R2 score: 0.87
Input is: [380, 381, 382, 385, 391]
Predicted CO2: 395.67
___________________________________________
Input is: [381, 382, 385, 391, 395.67]
Predicted CO2: 396.12
___________________________________________
...
```

### Plotting (optional):
You can uncomment the plotting sections to visualize the data and predictions as well:

```python
# plt.plot(data.time[:int(len(x)*train_ratio)], data["co2"][:int(len(x)*train_ratio)], label="train")
# plt.plot(data.time[int(len(x)*train_ratio):], data["co2"][int(len(x)*train_ratio):], label="test")
# plt.plot(data.time[int(len(x)*train_ratio):], y_predicted, label="predicted")
# plt.grid()
# plt.legend()
# plt.show()
```

## 📝 Evaluation Metrics
The model is evaluated using the following metrics:
- **MAE (Mean Absolute Error)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction.
- **MSE (Mean Squared Error)**: Measures the average squared difference between the actual and predicted values.
- **R² Score**: Indicates how well the model explains the variance in the target variable. A value close to 1 indicates a good fit.

## 🤖 Next Steps
- **Hyperparameter Tuning**: Experiment with different window sizes for lag features to optimize model performance.
- **Model Improvement**: Explore more advanced models like **Random Forests** or **Neural Networks** to further improve predictions.
- **Real-Time Data**: Integrate this system into a real-time environment where the model can continuously predict CO₂ levels based on streaming data.

## 📚 References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

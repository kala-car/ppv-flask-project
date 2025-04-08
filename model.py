import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv('Cleaned_Blasting_Dataset_Final.csv')
X = df.drop(columns=['PPV MON', 'TIME(Hrs)'])
y = df['PPV MON']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Build ANN model
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Evaluate and save
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model Accuracy (R² Score): {r2 * 100:.2f}%")
model.save('ppv_ann_model.h5')

print("Model and scaler saved successfully.")

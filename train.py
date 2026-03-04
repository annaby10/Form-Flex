import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Loading dataset...")
df = pd.read_csv('squat_coords_merged.csv')

# The CSV has exactly 133 columns: 'class' + 132 features (x1, y1, z1, v1... x33, y33, z33, v33)
X = df.drop('class', axis=1) # features
y = df['class'] # target

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Classes found: {y.unique()}")

print("Training Random Forest Classifier on Squat Dataset...")
# Train the RF model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Training complete! Model accuracy on training data: {:.2f}%".format(model.score(X, y) * 100))

# Save the model
with open('squat.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to squat.pkl successfully!")

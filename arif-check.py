import pickle
import numpy as np
print("check")

with open('classifier_rf.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print("Model loaded successfully")

y_pred = loaded_model.predict(np.array([[4, 2, 4, 0]]))

print(y_pred)
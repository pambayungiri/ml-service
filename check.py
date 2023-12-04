import pickle

print("check")

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print("Model loaded successfully")
x_test = [ 1.07790138,  0.14882364,  0.48043163, -0.81637123, -1.19509609,  0.98577893,
  1.0856502,   0.5059087,   0.53752595,  0.74109818]
y_pred = loaded_model.predict([x_test])

print(y_pred)
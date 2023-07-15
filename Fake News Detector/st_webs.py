import numpy as np
import pickle

loaded_model = pickle.load(open("C:/Users/91623/Fake News Detector/trained_model.sav",'rb'))

# prediction system

input_data = np.asarray(10)
inp_data_reshaped = input_data.reshape(-1,1)
prediction = loaded_model.predict(input_data)
if prediction[0] == 1:
    print("Fake News")
else:
    print("Real News")
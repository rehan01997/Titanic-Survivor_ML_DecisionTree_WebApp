import numpy as np
import pandas as pd
from flask import Flask , jsonify , render_template , request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl' , 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods = ['POST'])
def predict():
    #for rendering results on HTML GUI
    user_features = [int(x) for x in request.form.values()]
    #user_features = [1 ,0 ,21.0 ,0 ,0 ,77.9583]
    final_features = np.array(user_features)
    final_features = final_features.reshape(( 1 ,-1 ))
    final_features = pd.DataFrame(final_features)
    
    # print(final_features.head())
    output = model.predict(final_features)
    output = output[0]
    
    if( output == 0):
        final_output = "Dead"
    else:
        final_output = "Survived"
    
    return render_template('index.html' , prediction_text='The person will {}'.format(final_output))

if __name__ == "__main__":
    app.run(debug = True)

import json
import numpy as np
import os
import pickle
#from sklearn.externals import joblib
import joblib
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model

def init():
    global model
    try:
        folder=os.getenv('AZUREML_MODEL_DIR')
        print('folder=', folder)
        model_path = Model.get_model_path(os.path.join(folder,'model_alpha_0.1.pkl'))
        print('>>>>>Model path==',model_path)
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print(Model.get_model_path(model_name='model_alpha_0.1.pkl'))

        #retrieve the path to the model file using the model name
        #model_path = Model.get_model_path('model_alpha_0.1.pkl')   
        with open(model_path, 'rb') as file:
            model = joblib.load(file)

    except Exception as e:
       result= str(e)
       return json.dumps({'error':result})

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        # make prediction
        y_hat = model.predict(data)
        # you can return any data type as long as it is JSON-serializable
        return y_hat.tolist()
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return json.dumps({"error": result})

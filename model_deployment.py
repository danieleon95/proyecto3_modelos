#!/usr/bin/python
from sklearn.externals import joblib
import sys
import os

def predict(plot):
    mod = joblib.load(os.path.dirname(__file__) + '/model.pkl')
    vect = joblib.load(os.path.dirname(__file__) + '/vect.pkl')
    X_test = vect.transform(plot)
    y_pred = mod.predict_proba(X_test)
    d = {'p_Action': y_pred[0][1], 'p_Adventure': y_pred[0][1], 'p_Animation': y_pred[0][2], 'p_Biography': y_pred[0][3],
         'p_Comedy': y_pred[0][4], 'p_Crime': y_pred[0][5], 'p_Documentary': y_pred[0][6], 'p_Drama': y_pred[0][7], 'p_Family': y_pred[0][8],
         'p_Fantasy': y_pred[0][9], 'p_Film-Noir': y_pred[0][10], 'p_History': y_pred[0][11], 'p_Horror': y_pred[0][12], 'p_Music': y_pred[0][13],
         'p_Musical': y_pred[0][14], 'p_Mystery': y_pred[0][15], 'p_News': y_pred[0][16], 'p_Romance': y_pred[0][17], 'p_Sci-Fi':y_pred[0][18],
         'p_Short': y_pred[0][19], 'p_Sport': y_pred[0][20], 'p_Thriller': y_pred[0][21], 'p_War': y_pred[0][22], 'p_Western': y_pred[0][23]}
    return d

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add information')
        
    else:

        p1 = predict(sys.argv[1])
        
        print('result: ', p1)

from flask import Flask, request, render_template, jsonify, redirect
from flask.helpers import url_for
import pickle
import category_encoders as ce
import pandas as pd

app = Flask(__name__)

data = pd.read_csv('14Sept.csv')


def encode(data):
    encoder_1 = ce.OrdinalEncoder(cols=['Outlet_Size'], return_df=True,
                                  mapping=[{'col': 'Outlet_Size', 'mapping': {'Small': 0, 'Medium': 1, 'High': 2}}])
    encoder_2 = ce.OrdinalEncoder(cols=['Outlet_Location_Type'], return_df=True,
                                  mapping=[{'col': 'Outlet_Location_Type',
                                            'mapping': {'Tier 3': 0, 'Tier 2': 1, 'Tier 1': 2}}])

    data_transformed = encoder_1.fit_transform(data)
    data_transformed = encoder_2.fit_transform(data_transformed)
    data_encoded = pd.get_dummies(data_transformed, columns=['Item_Fat_Content', 'Outlet_Type', 'Item_Type_Combined',
                                                             'Item_Type'], drop_first=True)
    return data_encoded


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/result/<sales>')
def result(sales):
    return render_template('predictor.html', sales=sales)


@app.route('/predictor', methods=['POST', 'GET'])
def prediction():
    if request.method == 'GET':
        return render_template('predictor.html')
    elif request.method == 'POST':
        iWeight = float(request.form['weight'])
        iVisibility = float(request.form['visibility'])
        iFat = request.form['fat-content']
        iTypeCombined = request.form['item-type-combined']
        iType = request.form['item-type']
        iMrp = float(request.form['mrp'])
        oYear = request.form['outlet-year']
        oSize = request.form['outlet-size']
        oLocation = request.form['outlet-loc-type']
        oType = request.form['outlet-type']

        testInput = dict(iWeight=iWeight, iVisibility=iVisibility, iFat=iFat, iTypeCombined=iTypeCombined, 
        iType=iType, iMrp=iMrp, oYear=oYear, oSize=oSize, oLocation=oLocation, oType=oType)

        data.loc[len(data.index)] = [iWeight, iFat, iVisibility, iType, iMrp, oSize, oLocation, oType, iTypeCombined,
                                     oYear]
        df = encode(data)
        test = df.tail(1)

        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))

        predict = loaded_model.predict(test)
        testOutput = predict[0]

        output = dict(testInput=testInput, testOutput=testOutput)
        # return redirect(url_for('result', sales=output))
        return render_template('result.html', output=output)
    else:
        return jsonify({'replyMessage': "OOPS! Something gone wrong..."})


if __name__ == '__main__':
    app.run(debug=True)

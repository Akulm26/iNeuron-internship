# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:04:52 2021

@author: ayesha
"""


# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import cross_origin
import pickle
import category_encoders as ce
import pandas as pd

data=pd.read_csv('14Sept.csv')
data.columns


app = Flask(__name__, static_url_path='/static') # initializing a flask app


def encod(data):
    encoder_1= ce.OrdinalEncoder(cols=['Outlet_Size'],return_df=True,mapping=[{'col':'Outlet_Size','mapping':{'Small':0,'Medium':1,'High':2}}])
    encoder_2= ce.OrdinalEncoder(cols=['Outlet_Location_Type'],return_df=True,mapping=[{'col':'Outlet_Location_Type','mapping':{'Tier 3':0,'Tier 2':1,'Tier 1':2}}])
    #fit and transform train data 
    data_transformed = encoder_1.fit_transform(data)
    data_transformed = encoder_2.fit_transform(data_transformed)
    data_encoded=pd.get_dummies(data_transformed, columns = ['Item_Fat_Content','Outlet_Type','Item_Type_Combined', 'Item_Type'], drop_first=True)
    return data_encoded

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")




@app.route('/via_postman', methods=['POST']) # for calling the API from Postman/SOAPUI
def prediction_via_postman():
    if (request.method=='POST'):
        Item_Weight=float(request.json['Item_Weight'])
        Item_Visibility=float(request.json['Item_Visibility'])
        Item_MRP = float(request.json['Item_MRP'])    
        Outlet_Size =request.json['Outlet_Size']
        Outlet_Location_Type = request.json['Outlet_Location_Type']
        Years_Established = request.json['Years_Established']
        Item_Fat_Content = request.json['Item_Fat_Content']
        Outlet_Type = request.json['Outlet_Type']
        Item_Type_Combined = request.json['Item_Type_Combined']
        Item_Type = request.json['Item_Type']
       # print(Item_Weight)
        data.loc[len(data.index)] = [Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Size, Outlet_Location_Type, Outlet_Type, Item_Type_Combined, Years_Established]
        df=encod(data)
        test=df.tail(1)
        print(len(test.columns))
        print(test.columns)
        #test = test.drop(['Item_Type_Combined_Drinks'], axis = 1)
        
        
        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
        
        
        prediction=loaded_model.predict(test)
        
    
        #print(Item_Weight, Item_Visibility, Item_MRP, Outlet_Size, Outlet_Location_Type, Years_Established,
        #                Item_Fat_Content, Outlet_Type, Item_Type_Combined, Item_Type)
        return jsonify({'prediction': list(prediction)})

        

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001, debug=True, use_reloader=False)
	#app.run(debug=True) # running the app
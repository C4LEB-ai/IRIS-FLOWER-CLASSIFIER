# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from gettext import install
import streamlit as st
import pandas as pd
import numpy as np
import pickle

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "welcome all"

def predict_IRIS_Flower_Type(sepal_length, sepal_width, petal_length, petal_width):
    """ Lets predict the flower type 
    this is using the doctstrings for specifications.
    ---
    parameters:
        -name : sepal_length
        in: query
        type: number
        required : true
        -name: sepal_width
        in: querry
        type: number
        required: true
        -name: petal_length 
        in: query
        type: number
        required: true
        -name: petal_width 
        in: query
        type: number
        required: true
    responses:
        200:
            description: the output value
            
    """

    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction
    
    return"Hello The answer is"+str(prediction)
def main():
    st.title("IRIS flower Prediction")
    html_temp = """
    <div style = "background-color:tomato;padding:10px">
    <h2 style ="color:white;test-align:center;"> C4LEB IRIS flower classifier ML app</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sepal_length = st.text_input("sepal_length")
    sepal_width = st.text_input("sepal_width")
    petal_length = st.text_input("petal_length")
    petal_width = st.text_input("petal_width")
    result = ""
    if st.button("predict"):
        result = predict_IRIS_Flower_Type(sepal_length, sepal_width, petal_length, petal_width)
    st.success("the output is: {}".format(result))
    if st.button("about"):
        st.text(("This is an machine learnig application "))
        st.text("build for the classification of IRIS flowers based on labels provided above")

if __name__ == "__main__":
    main()
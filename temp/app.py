import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("/content/drive/My Drive/knnmodelnew.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('ishika PIET18CS065 - Classification Dataset1.csv')
X = dataset.iloc[:,5:10].values



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict_note_authentication(CreditScore, Geography, Gender, Age, Tenure, Balance, HasCrCard, IsActiveMember, EstimatedSalary):
  output= model.predict(sc.transform([[CreditScore, Geography, Gender, Age, Tenure, Balance, HasCrCard, IsActiveMember, EstimatedSalary]]))
  print("Model has predicted",output)
  if output==[0]:
    prediction="Customer will leave..."
   

  if output==[1]:
    prediction="Customert will continue..."
    
    
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:black;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Experiment Deployment By Ishika Jain</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer continue or not...")
    Name = st.text_input('Enter name')
    CreditScore = st.number_input('Insert credit score',100,1000)
    Geography= st.number_input('Insert 0 for France 1 for Spain',0,1)
    Gender = st.number_input('Insert 0 for Male 1 for Female ',0,1)
    age = st.number_input('Insert a Age',18,70)
    Tenure = st.number_input('Insert Tenure',0,40)
    Balance = st.number_input('Insert Balance',0,130000)
    HasCrCard = st.number_input('Insert 0 for no 1 for yes for Credit Card',0,1)
    IsActiveMember = st.number_input('Insert 0 for no 1 for yes for active member',0,1)
    EstimatedSalary = st.number_input('InsertEstimated salery ',0,150000)
    
    result=""
    if st.button("Predict"):
      result=predict_note_authentication(age,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Ishika Jain")
      st.subheader("B-Section,PIET")

if __name__=='__main__':
  main()
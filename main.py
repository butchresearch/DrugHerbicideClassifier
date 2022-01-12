import streamlit as st
import pandas as pd
from streamlit.type_util import data_frame_to_bytes
from st_aggrid import AgGrid
from dhclassifier import DHClassifier
dataframe  = pd.DataFrame()
user_input = None
st.title('Drug Herbicide Classifier ') #Set Tittle
st.write(
        """
        ## Explore our Models Online
        
        """)


#### SELECTION OF INPUT TYPE ####
InputSelector = st.sidebar.selectbox("Select Input Type",("Smiles String","CSV"))                     # Select input Type 
if InputSelector == "Smiles String":
        user_input         = st.sidebar.text_input("Select a Valid Smiles String:", "CC")             # Siles String INput
    
elif InputSelector == "CSV":
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
                user_input = pd.read_csv(uploaded_file,header=None)  # Convert csv to panda dataframe
                user_input = user_input.iloc[:, 0]                   # Get all the elements of the first column
                user_input = user_input.tolist()                     # Convert it to list 
         
### Select Visualization tool 
VisualizationSelector = st.sidebar.radio("Show Molecule",('Yes', 'No'))

OutputSelector = st.sidebar.selectbox("Select Outpuy Type",("Verbose","Simplified"))                     # Select input Type 



##### Work with dataframe ###
if user_input != None:
        dataframe = DHClassifier(user_input)
        if OutputSelector == "Verbose":
                pass
        else:
                dataframe =  dataframe[["SMILES","mol","XG_Drug","XG_Herbicide","LR_Drug","LR_Herbicide","RF_Drug","RF_Herbicide","SVM_Drug","SVM_Herbicide"]]


        if VisualizationSelector == "Yes":
                #st.dataframe(dataframe)
                st.write(dataframe.to_html(escape=False), unsafe_allow_html=True)
        elif VisualizationSelector == "No":
                try:
                        dataframe = dataframe.drop(columns=['mol'])
                except:
                        pass
                AgGrid(dataframe)

      

        ### Download  Dataframe###
        @st.cache
        def convert_df(df):
                return df.to_csv().encode('utf-8')



        csv = convert_df(dataframe)

        st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )
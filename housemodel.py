
import streamlit as st
import pandas as pd
import altair as alt
import keras
import pickle as pkl
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import os, datetime
from keras.utils.vis_utils import plot_model
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

df = pd.read_csv('C:\\Users\\shama\\Documents\\excelr_project\\Property_info1.csv')
#option at the side bar
analysis = st.sidebar.selectbox('Select an Option',['Price Prediction','Data Analysis & Visualization'])
# analysis
#title
st.set_option('deprecation.showfileUploaderEncoding', False)
if analysis=='Data Analysis & Visualization':
    st.header('Exploratory Data Analysis')
    # simple description
    st.write('In this dashboard we will analyze the house data set')
    st.image(Image.open(os.path.join('C:\\Users\\shama\\Downloads\\eda.jfif')))
    st.write('')
    
    # 2 types of heading - header and subheader
    st.markdown('Before we get started with the analysis, lets have a quick look at the raw data :sunglasses:')
    
    #show dataset
    if st.checkbox("Preview Dataset"):
        if st.button("Head"):
            st.write(df.head())
        elif st.button("Tail"):
              st.write(df.tail())
    
    if st.checkbox("Show All Dataset"):
        #st.write(df)
        st.dataframe(df)
    # show columns  
    if st.checkbox("Show Column names"):
        st.write(df.columns)
        
#show dimensions 
    data_dim = st.radio("What Dimensions do you want to see?",("Rows","Columns","All"))
    if data_dim == "Rows":
        st.text("Showing Rows")
        st.write(df.shape[0])
    elif data_dim == "Columns":
        st.text("Showing Columns")
        st.write(df.shape[1])
    else:
        st.text("ShowingShape of Dataset")
        st.write(df.shape)
    
    #Show summary
    if st.checkbox("Show Summary of Dataset"):
        st.write(df.describe())
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Corelation
    if st.checkbox("Show Correlation plot with Matplotlib"):
        plt.matshow(df.corr())
        st.pyplot()
        
    if st.checkbox("Show Correlation plot with Seaborn"):
        st.write(sns.heatmap(df.corr()))
        st.pyplot()
        
    fn,ax=plt.subplots(1,2,figsize=(10,5)) # Defining plot size and subplots.
    fn.suptitle("Location vs Count of properties in that loctaion") # Title for the plot.
    df.location.value_counts().head(30).plot(kind='barh',ax=ax[0]) # Plot for top 20 values.
    df.location.value_counts().tail(30).plot(kind='bar',ax=ax[1]) # Plot for least 20 values.
    st.pyplot(fn)
     
    fig1, ax1 = plt.subplots()
    fig1.suptitle("price counts") # Plot title
    ax1.hist(df.price)
    #ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)
    
    f,ax=plt.subplots(figsize=(5,7)) # Plot size
    ax2=plt.subplot(211) 
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    df['price'].value_counts(ascending=False).head(10).plot(kind='pie',autopct="%0.2f%%",ax=ax2) 
    plt.title("Price Percentages") # Plot title
    st.pyplot(f)
    
    f,ax=plt.subplots(figsize=(10,9)) # Plot size
    ax3=plt.subplot(224)
    sns.distplot(df['price'],ax=ax3) # Distribution plot for all prices.
    plt.title("Price Distribution") # Plot title
    st.pyplot(f)
    
    fig,ax=plt.subplots(figsize=(8,5))
    sns.countplot(df['area_type'],ax=ax) # Count plot
    plt.title("Area of property") # Plot title
    st.pyplot(fig)
    
    
    with st.sidebar.header('1. CSV data'):
        uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"])
        

    # Pandas Profiling Report
    if uploaded_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
        df = load_csv()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
    else:
        if st.button('Press to use Example Dataset'):
            # Example data
            @st.cache
            def load_data():
                a = pd.DataFrame(
                    np.random.rand(100, 5),
                    columns=['a', 'b', 'c', 'd', 'e']
                )
                return a
            df = load_data()
            pr = ProfileReport(df, explorative=True)
            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)
            
            st.graphviz_chart('''
            digraph {
                run -> intr
                intr -> runbl
                runbl -> run
                run -> kernel
                kernel -> zombie
                kernel -> sleep
                kernel -> runmem
                sleep -> swap
                swap -> runswap
                runswap -> new
                runswap -> runmem
                new -> runmem
                sleep -> runmem
            }
        ''')
            
        
else:
    pickle_in = open('C:\\Users\\shama\\Desktop\\model.pickle','rb')
    classifier = pkl.load(pickle_in)


    def Welcome():
        return 'WELCOME ALL!'
    
    st.date_input("Today date",datetime.datetime.now())
    
    def predict_price(location,sqft,bath,bhk,area):    
        """Let's Authenticate the Banks Note 
        This is using docstrings for specifications.
        ---
        parameters:  
          - name: location
            in: query
            type: text
            required: true
          - name: sqft
            in: query
            type: number
            required: true
          - name: bath
            in: query
            type: number
            required: true
          - name: area
            in: query
            type: number
            required: true
        responses:
            200:
                description: The output values
            
        """
        #loc_index = np.where(X.columns==location)[0][0]

        x = np.zeros(5)
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        #x[3] = area
        #if loc_index >= 0:
            #   x[loc_index] = 1

        return classifier.predict([x])[0]

    def main():
        st.title("Welcome to Bangalore House Price Prediction")
        html_temp = """
        <h2 style="color:black;text-align:left;"> Streamlit House prediction ML App </h2>
        """
        st.image(Image.open(os.path.join('C:\\Users\\shama\\Downloads\\image.jfif')))
        st.markdown(html_temp,unsafe_allow_html=True)
        st.subheader('Please enter the required details:')
        loc_name = st.selectbox('Select a Location', options=df.location.unique())
        #location = st.text_input("Location","")
        sqft = st.text_input("Sq-ft area","")
        bath = st.text_input("Number of Bathroom","")
        bhk = st.text_input("Number of BHK","")
        area = st.selectbox('Select Area Type', options=df.area_type.unique())
        result=""

        

        if st.button("House Price in Lakhs"):
            result=predict_price(loc_name,sqft,bath,bhk,area)
        st.success('The output is {}'.format(result))
        
           

    if __name__=='__main__':
        main()
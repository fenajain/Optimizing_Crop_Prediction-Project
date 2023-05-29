import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from opt import *
import streamlit as st

st.set_option('browser.gatherUsageStats', False)

st.title("Crop Prediction System")
df = pd.read_csv("CropData.csv")

input_data = []

N = st.text_input("Enter the Nitrogen Content (1-150) ")
if N:
    try:
        Nitrogen = float(N)
        if Nitrogen < 1 or Nitrogen > 150 :
            st.error("Nitrogen content should be between 1 to 150.")
        else:
            input_data.append(Nitrogen)

    except ValueError:
        st.error("Invalid Input, Please Enter a Numeric Value.")

P = st.text_input("Enter the Phosphorus Content (1-150) ")
if P:
    try:
        Phosphorus = float(P)
        if Phosphorus < 1 or Phosphorus > 150:
            st.error("Phosphorus content should be between 1 to 150.")
        else:
            input_data.append(Phosphorus)

    except ValueError:
        st.error("Invalid Input, Please Enter a Numeric Value.")

K = st.text_input("Enter the Potassium Content (1-200) ")
if K:
    try:
        Potassium = float(K)
        if Potassium < 1 or Potassium > 200:
            st.error("Potassium content should be between 1 to 200.")
        else:
            input_data.append(Potassium)

    except ValueError:
        st.error("Invalid Input, Please Enter a Numeric Value.")

temp = st.text_input("Enter the Temperature (1-50) ")
if temp:
    try:
        Temperature = float(temp)
        if Temperature < 1 or Temperature > 50:
            st.error("Temperature should be between 1 to 50.")
        else:
            input_data.append(Temperature)

    except ValueError:
        st.error("Invalid Input, Please Enter a Numeric Value.")

hum = st.text_input("Enter the Humidity Value (10-100) ")
if hum:
    try:
        Humidity = float(hum)
        if Humidity < 10 or Humidity > 100:
            st.error("Humidity content should be between 10 to 100.")
        else:
            input_data.append(Humidity)

    except ValueError:
        st.error("Invalid Input, Please Enter a Numeric Value.")

ph = st.text_input("Enter the pH Content (3-10) ")
if ph:
    try:
       pH = float(ph)
       if pH < 3 or pH > 10:
           st.error("pH content should be between 3 to 10.")
       else:
           input_data.append(pH)

    except ValueError:
        st.error("Invalid Input, Please Enter a Numeric Value.")

rf = st.text_input("Enter the Rainfall Value (20-300) ")
if rf:
    try:
        Rainfall = float(rf)
        if Rainfall < 20 or Rainfall > 300:
            st.error("Rainfall content should be between 20 to 300.")
        else:
            input_data.append(Rainfall)

    except ValueError:
        st.error("Invalid Input, Please Enter a Numeric Value.")

result = ""

def crop_prediction(ip_data):
    input_array = np.asarray(ip_data)
    reshaped_data = input_array.reshape(1, -1)
    num_data = reshaped_data.astype(np.float64)        # The value received from above is in string form so to convert that into string we did this
    return num_data


try:
    if st.button("Predict & Show Analysis"):
        final = crop_prediction(input_data)
        result = model.predict(final)
        res = np.array(result)
        st.write("<span style = 'font-size : 24px'> The output is <u><b>{}</b></u></span>".format(res[0].capitalize()),unsafe_allow_html = True)

        crop_name = res[0]
        seasonal_crop = suggest_season(crop_name, df)
        st.write("<span style = 'font-size : 24px'> Your crop {} is <u><b>{}</b></u> crop</span>".format(res[0],seasonal_crop.capitalize()),unsafe_allow_html = True)

        stats = summary(crop_name)
        st.write(stats)

        
        data_frame = summary(crop_name)
        st.markdown("<h1 style='text-align: left; font-size: 30px;'>Different Quantity of Chemical Content in {} crop</h1>".format(res[0].capitalize()), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.lineplot(x="Labels", y="Minimum", data=data_frame, ax=ax , color = "red" , label = "Minimum Content")
        sns.lineplot(x="Labels", y="Average", data=data_frame, ax=ax , color = "blue" , label = "Average Content")
        sns.lineplot(x="Labels", y="Maximum", data=data_frame, ax=ax, color="green" , label = "Maximum Content")
        # Display plot on Streamlit
        ax.legend(loc='upper left')
        ax.set_xlabel("Chemical Components in the Soil",fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_ylabel('Quantity of Chemical Components',fontsize = 10)
        ax.tick_params(axis='y', labelsize=8)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error occurred is: {e}")

try:

    if st.button("Data_Analytics"):
        frame = pd.read_csv("detailed_cropdata.csv")

        st.markdown("<h1 style='text-align: left; font-size: 30px;'>Overall Data Visualization </h1>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,2))
        # plt.figure(figsize=(5,2))
        ax.bar(frame['Crop_Labels'], frame['N-Avg'], color='thistle')
        plt.xticks(rotation=90)
        # Set plot title and axis labels
        ax.set_title('Different Crops VS Average Nitrogen Content',fontsize = 10)
        ax.set_xlabel('Different Crops', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_ylabel('Average Nitrogen Content', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        st.pyplot(fig)


        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
        ax1.bar(frame['Crop_Labels'], frame['P-Avg'], color='khaki')
        ax1.set_title('Different Crops VS Average Phosphorus Content', fontsize=12)
        ax1.set_xlabel('Different Crops', fontsize=10)
        ax1.tick_params(axis='x', labelsize=7)
        ax1.set_ylabel('Average Phosphorus Content', fontsize=10)
        ax1.tick_params(axis='x', labelsize=7 , rotation = 90)

        sns.kdeplot(frame['P-Avg'], shade=True, color='slateblue', ax=ax2)
        ax2.set_title('Distribution of Average Phosphorus Content', fontsize=12)
        ax2.set_xlabel('Average Phosphorus Content', fontsize=10)
        ax2.tick_params(axis='x', labelsize=7)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.tick_params(axis='x', labelsize=7)
        st.pyplot(fig)


        fig, ax = plt.subplots(figsize=(5, 2))
        # Sort the data
        data = sorted(frame['K-Avg'])
        # Create histogram plot
        ax.hist(data, color='rosybrown')
        # Set plot title and axis labels
        ax.set_title('Distribution of Average Potassium Content', fontsize=10)
        ax.set_xlabel('Potassium Content in the soil', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        st.pyplot(fig)


        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x="Crop_Labels", y="T-Avg", data=frame, ax=ax, color="hotpink", label="Avg. Temp")
        sns.lineplot(x="Crop_Labels", y="H-Avg", data=frame, ax=ax, color="cornflowerblue", label="Avg. Humidity")
        sns.lineplot(x="Crop_Labels", y="Rf-Avg", data=frame, ax=ax, color="brown", label="Avg. Rainfall")
        plt.xticks(rotation=90)
        ax.legend(loc='upper right')
        ax.set_xlabel("Chemical Components in the Soil", fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_ylabel('Quantity of Chemical Components', fontsize=10)
        ax.tick_params(axis='y', labelsize=8)
        st.pyplot(fig)


        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
        ax1.bar(frame['Crop_Labels'], frame['ph-Avg'], color="burlywood")
        ax1.set_title('Different Crops VS Average pH Content', fontsize=12)
        ax1.set_xlabel('Different Crops', fontsize=10)
        ax1.tick_params(axis='x', labelsize=7)
        ax1.set_ylabel('Average pH Content', fontsize=10)
        ax1.tick_params(axis='x', labelsize=7, rotation=90)

        sns.kdeplot(frame['ph-Avg'], shade=True, color='orchid', ax=ax2)
        ax2.set_title('Distribution of Average pH Content', fontsize=12)
        ax2.set_xlabel('Average pH Content', fontsize=10)
        ax2.tick_params(axis='x', labelsize=7)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.tick_params(axis='x', labelsize=7)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error occurred is: {e}")

try:
    st.markdown("<h1 style='text-align: left; font-size: 24px;'>Some Interesting Facts to known about crops</h1>",unsafe_allow_html = True)
    if st.button("Facts") :

        output = knowing_facts(df)

        st.write("1. {} requires very high ratio of Nitrogen content in soil.".format(output[0].capitalize()))
        st.write("2. {} and {} requires very high ratio of Phosphorus content in soil.".format(output[1].capitalize(),output[2].capitalize()))
        st.write("3. {} and {} requires very high ratio of Potassium content in soil.".format(output[3].capitalize(), output[4].capitalize()))
        st.write("4. {}, {} and {} requires very high ratio of Rainfall.".format(output[5].capitalize(), output[6].capitalize(), output[7].capitalize()))
        st.write("5. {} requires very low Temperature content.".format(output[8].capitalize()))
        st.write("6. {} requires very high Temperature content in soil.".format(output[10].capitalize()))
        st.write("7. {} and {} requires very low ratio of Humidity content in soil.".format(output[11].capitalize(), output[12].capitalize()))
        st.write("8. {} requires very both high and low pH content ranging between 3.5 to 10".format(output[13].capitalize()))

except Exception as e:
    st.error(f"Error occurred is: {e}")








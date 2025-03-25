import streamlit as st
import pandas as pd
import numpy as np
import run_model
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def plot_history_streamlit(history):
    # Convert the history object to a DataFrame for easy plotting
    hist = pd.DataFrame(history.history)
    # If history.epoch is available, add it as a column
    hist['epoch'] = history.epoch

    # Plot Mean Absolute Error
    #fig1, ax1 = plt.subplots()
    #ax1.set_xlabel('Epoch')
    #ax1.set_ylabel('Mean Abs Error [MPG]')
    #ax1.plot(hist['epoch'], hist['mae'], label='Train Error')
    #ax1.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    #ax1.legend()
    #ax1.set_ylim([0, 3000])
    #st.pyplot(fig1)

    # Plot Mean Square Error
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Square Error [$MPG^2$]')
    ax2.plot(hist['epoch'], hist['mse'], label='Train Error')
    ax2.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    ax2.legend()
    #st.pyplot(fig2)

    fig2_zoom, ax2_zoom = plt.subplots()
    ax2_zoom.set_xlabel('Epoch')
    ax2_zoom.set_ylabel('Mean Square Error [$MPG^2$]')
    ax2_zoom.plot(hist['epoch'], hist['mse'], label='Train Error')
    ax2_zoom.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    ax2_zoom.legend()
    ax2_zoom.set_ylim([0, 5])
    #st.pyplot(fig2_zoom)

    col1, col2 = st.columns(2)

    with col1: 
        st.pyplot(fig2)
    with col2:
        st.pyplot(fig2_zoom)

@st.cache_resource
def plot_predictions_vs_labels(y_true, y_pred):
    """
    Plots two curves: one for true labels and one for predictions.
    Assumes there are 24 data points in each array.

    Parameters:
    -----------
    y_true : array-like of shape (24,)
        The true label values.
    y_pred : array-like of shape (24,)
        The predicted values.
    """
    # Create an x-axis array representing each data point index (0 to 23)
    x = np.arange(len(y_true))

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()

    # Plot true labels as a line with markers
    ax.plot(x, y_true, marker='o', linestyle='-', color='blue', label='True Labels')

    # Plot predictions as a line with markers
    ax.plot(x, y_pred, marker='o', linestyle='-', color='red', label='Predictions')

    # Labeling the axes and the plot
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel("Value")
    ax.set_title("Test Predictions vs. True Labels")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

@st.cache_resource
def plot_data(df):
    for col in df.columns:
        fig, ax = plt.subplots()
        sns.scatterplot( data=df, x=col, y='Load', ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Load')
        st.pyplot(fig)

# predictions: 
#@st.cache_resource
def plot_backtest(train,predictions,labels):
    
    # Get copy of train features
    df_with_predictions = train.copy()

    # Split train features into each hour
    df_hours = []
    for i in range(24):
        df_hour = df_with_predictions[df_with_predictions.index.get_level_values('DateTime').hour == i].copy()
        #print(len(df_hour))
        #print(i)
        df_hours.append(df_hour)

    #print(f'Predictions: {len(predictions)}')
    i = 0
    for p in predictions:
        #print(f'Len: {len(p)}, Iter: {i}')
        i += 1

    # For each hour, merge the predictions to df
    for i in range(24):
        pred_h = predictions[i]
        df_hours[i].loc[:,'Prediction'] = pred_h
    
    # Concatinate df_hours into one
    df = pd.concat(df_hours)

    # Sort df by DateTime
    df = df.sort_index()
    
    df['True-Labels'] = labels

    st.write('Train: ', train)
    st.write('BackTest df: ',df)



    st.write('BackTest Plots')
   

    # Turn index into a feature.
    df['DateTime'] = df.index.get_level_values('DateTime')

 
 
    # DateTime vs Y (Interactive)
    fig = px.line(df,x='DateTime' ,y=['Prediction','True-Labels'],  title="Predictions vs. True Labels Over Time"
                  ,color_discrete_sequence=[ "orange", "blue"])
    st.plotly_chart(fig)

    # Scatter plot. 
    # Xi vs Y
    # Melt two load values
    df_melt = pd.melt(df, 
                  id_vars=[cols for cols in df.columns if cols != 'Prediction' and cols != 'True-Labels'], 
                  value_vars=['Prediction', 'True-Labels'], 
                  var_name='Type', 
                  value_name='Load')
    for col in df_melt.columns:
        # Only iterate through features.
        if col not in[ 'Load', 'Type', 'DateTime']:
            #x = df[col]
            p, ax = plt.subplots()
            sns.scatterplot(data=df_melt, x=col, y='Load', hue='Type', ax=ax)
            ax.set_xlabel(col)
            st.pyplot(p)

    
    # For each model
    st.write('BackTest Plots for each hour')
    h = st.slider(label='Model of Hour', min_value=0,max_value=23,value=0)
    model_of_hour = df_melt[df_melt['DateTime'].dt.hour == h]
    for col in model_of_hour.columns:
        # Only iterate through features.
        if col not in[ 'Load', 'Type', 'DateTime']:
            #x = df[col]
            p, ax = plt.subplots()
            sns.scatterplot(data=model_of_hour, x=col, y='Load', hue='Type', ax=ax)
            ax.set_xlabel(col)
            st.pyplot(p)




# Cache the training
@st.cache_resource
def get_model_results():
    results = run_model.ModelResults()
    results.run()
    print("Finished Training")
    hrs, rem = divmod(run_model.total_time, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Estimated Total Training Time: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}")
    
    return results


# Main Here:

st.title("Electricity Load Forecasting")

results = get_model_results()



df = results.df
train = results.train
train_labels = results.train_labels
test = results.test
test_labels = results.test_labels
models = results.models
histories = results.histories
test_predictions = results.test_predictions
average_error = results.average_error

# Plot Data
st.write("Data: ", df)
st.write('Train: ', train)
st.write('Train Labels: ', train_labels )
st.write('Test: ', test)
st.write('Test Labels: ', test_labels )
plot_data(df=df)

# Plot Train & Validation Errors
st.write('Train & Validation Errors')
selected_hour = st.slider('Select Hour' , 0,23, 0)
st.write("Hour: ", selected_hour)
plot_history_streamlit(history=histories[selected_hour])

# Plot Test Predictions
st.write('Test Predictions: ', test_predictions)
plot_predictions_vs_labels(y_true=test_labels,y_pred=test_predictions)
st.write('Average Error in MW: ', average_error)

#train_sample = train[-24:]
#train_sample_labels = train_labels[-24:]
#train_sample_predictions = model.predict(train_sample)

st.write('Training Backtest: ')
# Get the true and predictions from backtest. 
backtest_predictions = results.backtest_predictions

# Plot the backtest
plot_backtest(train=train, predictions=backtest_predictions,labels=train_labels)


#plot_predictions_vs_labels(y_true=train_sample_labels,y_pred=train_sample_predictions)








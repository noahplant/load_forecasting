# load_forecasting
Millennium Load Forecasting 

*Environment
- Make sure to create your own env 
- Make sure it can run streamlit
- If there is an error and it needs to run tensorflow (if you run NN) it could 
be a mac issue. 

Gather Data:
- Weather: Remove csv files to fetch the newest data (don't remove the hist file, because it takes ages to run)
- python get_load_data.py
- Load: Remove csv files to fetch the newest data (don't remove the hist file, because it takes ages to run)
- python get_weather_data.py

Run the 24 hour prediction 
- streamlit run run_model_tomorrow.py

Run the historical predictions
- streamlit run run_model.py

Go to the localhost to see the visualization and scroll down to get the prediction texts. 

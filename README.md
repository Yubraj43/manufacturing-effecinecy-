# Manufacturing Efficiency AI

## Overview
The Manufacturing Efficiency AI project aims to leverage machine learning techniques to analyze and improve manufacturing efficiency within the Thales Group. This project includes data analysis, model training, and a user-friendly application for visualizing results.

## Project Structure
```
Manufacturing_Efficiency_AI
├── data
│   └── Thales_Group_Manufacturing.csv
├── notebooks
│   └── eda_analysis.ipynb
├── models
│   └── efficiency_model.pkl
├── app
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## Files Description

- **data/Thales_Group_Manufacturing.csv**: Contains manufacturing data for the Thales Group, structured in a tabular format with various features relevant to manufacturing efficiency.

- **notebooks/eda_analysis.ipynb**: A Jupyter notebook for exploratory data analysis (EDA), including data visualization, statistical analysis, and insights derived from the manufacturing data.

- **models/efficiency_model.pkl**: A serialized machine learning model saved in the pickle format, used for predicting or analyzing manufacturing efficiency based on the trained model.

- **app/streamlit_app.py**: The main Python file for the Streamlit application, which loads the model, processes input data, and displays results or visualizations to the user.

- **requirements.txt**: Lists the Python dependencies required for the project, ensuring that the necessary libraries are installed for the application to run properly.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd Manufacturing_Efficiency_AI
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run app/streamlit_app.py
   ```

## Usage
Once the application is running, you can input manufacturing data and visualize the efficiency predictions based on the trained model. The EDA notebook can be used to explore the dataset and gain insights before model training.
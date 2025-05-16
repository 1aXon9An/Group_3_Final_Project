# How to run the Interactive Dashboard

This document provides instructions on how to run and view the interactive dashboard created for the **Cross Asset Financial Analysis and Forecasting of VNIndex, Gold and Bitcoin** project. 
Before running the dashboard, please ensure you have the following library installed:

* **Required Libraries:** If you haven't installed them yet, open your command prompt or terminal and run:
    ```bash
    pip install dash dash-bootstrap-components plotly pandas numpy scipy statsmodels pmdarima scikit-learn ta pandas_datareader
    ```
## Running the Dashboard
**🔴 IMPORTANT:** Please remember to change the data file path in both files before running the code file
There are two primary ways to run the dashboard:

### Using a Jupyter Notebook (.ipynb file)

Just download the file dashboard.ipynb and run
* **Limitation:** Running a Dash application directly within a standard Jupyter Notebook output cell might **not display the dashboard at its full size** or render the layout perfectly as intended. It's suitable for quick checks but not the ideal viewing experience.

### Option 2: Using the Python file (.py file)

Just running the .py file, and after that you will see a link to local web sever. Look for output in the terminal similar to this:
![image](https://github.com/user-attachments/assets/9fafcc47-c4fd-4ed9-84c4-b2c973b921f7)
* Copy this full URL (e.g., http://127.0.0.1:8050/).

After that open your preferred web browser then paste the copied URL into the browser's address bar and press Enter.

### The interface of the dashboard
![image](https://github.com/user-attachments/assets/da0cf356-88d7-4216-97c9-2724e30c7b61)
![image](https://github.com/user-attachments/assets/d047526f-3981-42f7-a4d2-53ac25847dec)
![image](https://github.com/user-attachments/assets/f672d929-9a86-4e3b-8844-b0bd524f1280)




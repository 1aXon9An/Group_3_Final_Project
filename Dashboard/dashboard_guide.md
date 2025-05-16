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

### Option 1: Using a Jupyter Notebook (.ipynb file)

Just download the file dashboard.ipynb and run
* **Limitation:** Running a Dash application directly within a standard Jupyter Notebook output cell might **not display the dashboard at its full size** or render the layout perfectly as intended. It's suitable for quick checks but not the ideal viewing experience.

### Option 2: Using the Python Script via Terminal (recommend)

This method runs the dashboard as a standalone web application, ensuring correct display and full functionality.

1.  **Open Command Prompt (CMD) or Terminal:**
    * Launch your system's terminal application

2.  **Navigate to the Dashboard Directory:**
    * Use the `cd` command to move into the folder containing the dashboard script (`dashboard.py`).
    * Example (replace with the actual path on your computer):
        ```bash
        cd D:\Study\3. CODE\1. Python_code_file\Big_Data\group_assignment\Dashboard
        ```
3.  **Run the Python Script:**
    * Execute the dashboard script by typing the following command and pressing Enter:
        ```bash
        python dashboard.py
        ```
4.  **Access the Dashboard URL:**
    * After the script runs successfully, it will start a local web server. Look for output in the terminal similar to this:
    ![image](https://github.com/user-attachments/assets/15c4d68c-6ed7-4a0a-8f64-79419c597253)
    * Copy this full URL (e.g., `http://127.0.0.1:8050/`).

5.  **View in Browser:**
    * Open your preferred web browser then paste the copied URL into the browser's address bar and press Enter.

6.  **Stop the Server:**
    * When you are finished, go back to the Command Prompt/Terminal window where the script is running.
    * Press `Ctrl + C` to shut down the local web server.
---
This method (Option 2) is recommended for the best user experience with the interactive features and layout of the Dash application. 

### The interface of the dashboard


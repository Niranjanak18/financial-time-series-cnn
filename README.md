Financial Time Series Analysis using STFT and CNN

Student Details
Name: Niranjana k 
Roll Number: TCR24CS053
Course: B.Tech Computer Science and Engineering

Project Overview
This project demonstrates how financial time series data (stock prices) can be analyzed as signals using signal processing and deep learning techniques. Stock price data is converted into a spectrogram using Short-Time Fourier Transform (STFT), and a Convolutional Neural Network (CNN) is used to learn patterns and predict future values.

Objectives
- Represent financial data as a signal
- Apply Short-Time Fourier Transform (STFT)
- Generate spectrogram for visualization
- Use CNN for prediction

Technologies Used
- Python 3.10
- NumPy
- SciPy
- Matplotlib
- TensorFlow / Keras
- yfinance

Methodology
1. Data Collection
Stock price data is collected using yfinance.

2. Signal Representation
Closing prices are normalized and treated as a signal.

3. STFT and Spectrogram
STFT is applied to convert the signal into time-frequency representation.

4. CNN Model
Spectrogram is used as input to a CNN model which learns patterns and predicts values.

Results
- Spectrogram reveals hidden patterns in stock data
- CNN model successfully learns from the data
- Prediction output is generated

Project Structure
financial_project/
 main.py
 README.md
 

How to Run
1. Activate environment
cnn_env\Scripts\activate

2. Run the program
python main.py

Conclusion
This project proves that financial time series data can be analyzed using signal processing techniques and deep learning models like CNN.

Author
Niranjana
Roll No: YOUR_ROLL_NO

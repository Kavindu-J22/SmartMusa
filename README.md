# SmartMusa - Banana Farming Market Recommendation System

SmartMusa is a comprehensive system designed to help banana farmers in Sri Lanka make informed decisions about when and where to sell their produce. The system uses machine learning to predict market prices and provide personalized recommendations.

## Features

- **Price Prediction**: Predicts banana prices based on historical data and current market trends
- **Market Recommendations**: Suggests optimal markets for selling based on location, quantity, and predicted prices
- **Multilingual Chatbot**: User-friendly interface that communicates with farmers in their native languages
- **Data-Driven Insights**: Provides insights based on historical market data analysis

## Project Structure

```
SmartMusa/
├── app.py                  # Main Flask application
├── chatbot.py              # Chatbot functionality
├── config.py               # Application configuration
├── data_processing.py      # Data processing and model training
├── train_model.py          # Script to train the prediction model
├── inspect_data.py         # Script to inspect Excel data
├── requirements.txt        # Required Python packages
├── data/                   # Directory for data files
│   └── weekly retail price of Ambulslit.xlsx  # Historical price data
└── static/                 # Static files for web interface (if needed)
```

## Setup Instructions

### Prerequisites

- Python 3.7+
- pip (Python package manager)
- Excel data file (weekly retail price of Ambulslit.xlsx)

### Installation

1. Clone the repository or download the source code

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   pip install seaborn
   ```

5. Make sure your Excel data file is in the correct location:
   - Place `weekly retail price of Ambulslit.xlsx` in the main directory or in a `data` subdirectory

### Training the Model

Run the following command to train the price prediction model:

```bash
python train_model.py
```

This will process the data, train the model, and save it as `banana_price_model.pkl`.

### Running the Application

Start the Flask application:

```bash
python app.py
```

The API will be available at `http://localhost:5000`.

## API Endpoints

- `GET /`: Home page
- `GET /api/v1/health`: Health check endpoint
- `POST /api/v1/predict`: Predict banana prices
- `POST /api/v1/recommend`: Get market recommendations
- `POST /api/v1/chatbot`: Interact with the chatbot
- `GET /api/v1/supported-languages`: Get supported languages
- `GET /api/v1/markets`: Get available markets
- `POST /api/v1/train`: Train or retrain the model

## Mobile App Integration (Expo)

To integrate with the Expo mobile app:

1. Make sure the Flask backend is running
2. Note the IP address of your computer (e.g., 192.168.1.100)
3. Update the API base URL in your Expo app's configuration to point to your Flask server
4. Build and run your Expo app

## Troubleshooting

### Common Issues

#### Excel File Not Found
- Ensure the file is named correctly: `weekly retail price of Ambulslit.xlsx`
- Try placing the file in both the root directory and a `data/` subdirectory

#### Model Training Fails
- Check your Excel file structure using `python inspect_data.py`
- Ensure your Excel file contains pricing data in a format that can be processed

#### API Connection Issues
- Verify the Flask server is running
- Check for any firewall or network restrictions
- Ensure the correct IP address and port are being used

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
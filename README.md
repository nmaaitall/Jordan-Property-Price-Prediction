# 🏢 Jordan Property Price Prediction

An AI-powered web application for real estate valuation in Jordan using Machine Learning. This system provides accurate property price estimates based on various features such as location, size, amenities, and property characteristics.

## 🌟 Features

- **Machine Learning Predictions**: Uses Random Forest algorithm for accurate price estimation
- **Bilingual Interface**: Full support for both Arabic and English languages
- **Theme Support**: Light and Dark mode for comfortable viewing
- **Interactive Visualizations**: Beautiful charts and graphs using Plotly
- **Real-time Analysis**: Instant property valuation with market comparison
- **User-Friendly Design**: Clean, modern, and responsive interface
- **Comprehensive Inputs**: 
  - Property location (20 regions in Jordan)
  - Area, bedrooms, bathrooms
  - Property age and floor number
  - Amenities (elevator, parking, garden, heating)
  - Proximity to services

## 🛠️ Tech Stack

- **Python 3.x**: Core programming language
- **Streamlit**: Web application framework
- **scikit-learn**: Machine Learning (Random Forest Regressor)
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualizations
- **NumPy**: Numerical computations

## 📊 Dataset

The dataset includes property data from 20 major regions in Jordan, including:
- Abdoun, Sweifieh, Khalda
- Shmeisani, Jubeiha, Sweileh
- And 14 other key areas

Features include property characteristics, amenities, and corresponding prices in Jordanian Dinars (JOD).

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nmaaitall/Jordan-Property-Price-Prediction.git
cd Jordan-Property-Price-Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run web_app.py
```

4. Open your browser and navigate to:
```
http://localhost:8501
```

## 📝 Usage

1. **Select Language and Theme**: Choose your preferred language (English/Arabic) and theme (Light/Dark) from the sidebar

2. **Enter Property Details**:
   - Select the region/location
   - Input property size (square meters)
   - Specify number of bedrooms and bathrooms
   - Enter property age and floor number

3. **Select Amenities**:
   - Check available features (elevator, parking, garden, heating)
   - Adjust proximity to services slider

4. **Get Prediction**:
   - Click "Calculate Property Value" button
   - View estimated price, market comparison, and analytics

## 🎯 Model Performance

The Random Forest Regressor model provides reliable predictions by considering:
- Property location and regional market trends
- Physical characteristics (size, rooms, age)
- Available amenities and features
- Proximity to essential services

## 📁 Project Structure

```
Jordan-Property-Price-Prediction/
├── web_app.py                    # Main Streamlit application
├── price_prediction_model.py     # ML model training script
├── data_analysis.py              # Data analysis utilities
├── generate_jordan_data.py       # Dataset generation script
├── jordan_properties.csv         # Property dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🔧 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.17.0
numpy>=1.24.0
```

## 🌐 Demo

[https://realpredict.streamlit.app]

## 📸 Screenshots

##.......

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👨‍💻 Author

**Nasser Maaitah**
- GitHub: [@nmaaitall](https://github.com/nmaaitall)
- LinkedIn: [https://jo.linkedin.com/in/nour-maaita-197733243]

## 🙏 Acknowledgments

- Thanks to the Streamlit community for the excellent framework
- Inspired by real estate market needs in Jordan
- Data science and ML community for continuous learning resources

## 📧 Contact

For questions or feedback, please open an issue or contact me directly.

---

**Note**: This project is for educational and demonstration purposes. Property valuations should be verified with professional real estate appraisers for actual transactions.

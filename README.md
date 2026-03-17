# 🌱 Plant Water Requirement Predictor

A **Machine Learning-powered web application** that predicts the **daily water requirement for plants** based on plant characteristics and environmental conditions.

🔗 **Live Demo:**
https://water-recommendation-4.onrender.com/

---

# 🚀 Features

* 🌿 Predicts **daily water requirement (Liters/day)**
* 🤖 Uses a trained **Machine Learning pipeline**
* 🌍 Web interface for easy interaction
* 🔌 REST API endpoint for external integrations
* ☁️ Deployed on cloud

---

# 🖥️ Live Application

You can access the deployed application here:

🔗 **https://water-recommendation-4.onrender.com/**

The app allows users to input:

* Plant Type
* Area (m²)
* Soil Type
* Climate
* Temperature

and receive the **estimated water requirement per day**.

---



# 📂 Project Structure

```
Water-Recommendation
│
├── app.py
├── requirements.txt
├── water_requirement_pipeline.joblib
├── train_model.py
├── predict.py
├── tropical_plants_2000_dataset.xlsx
│
├── templates/
│   └── index.html
│
├── static/
│
└── README.md
```

---

# ⚙️ Installation

### Clone the repository

```
git clone https://github.com/<username>/Water-Recommendation.git
cd Water-Recommendation
```

---

### Create virtual environment

```
python -m venv venv
```

Activate it

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

---

### Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Running the Application

```
python app.py
```

Open in browser

```
http://127.0.0.1:5000
```

---

# 📡 API Usage

### Endpoint

```
POST /predict
```

### Example Request

```
{
  "Plant": "Tomato",
  "Area_m2": 10,
  "Soil_Type": "Loamy",
  "Climate": "Tropical",
  "Temperature_C": 30
}
```

### Example Response

```
{
  "water_L_per_day": 15.27
}
```

---

# 🧠 Machine Learning Model

The model is saved as:

```
water_requirement_pipeline.joblib
```

It is built using **Scikit-Learn** and predicts water requirement using:

* Plant type
* Area
* Soil type
* Climate
* Temperature

---

# 🛠 Technologies Used

* Python
* Flask
* Scikit-Learn
* Pandas
* NumPy
* Joblib
* HTML / CSS

---

# ☁️ Deployment

The application is deployed on cloud using:

* Render
* Gunicorn
* Flask

Start command used:

```
gunicorn app:app
```

---

# 🔮 Future Improvements

* Add **real-time weather data integration**
* Support more plant species
* Smart irrigation scheduling
* Mobile responsive UI
* IoT irrigation integration

---

# 👨‍💻 Author

Developed as part of a **Machine Learning project for plant water prediction and irrigation optimization**.

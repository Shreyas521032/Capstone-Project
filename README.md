# Healthcare Data Analytics Platform

## Problem Statement
In modern healthcare, **data-driven decision-making** is essential for improving patient outcomes, enhancing user engagement, and optimizing healthcare resources. Despite the growing adoption of digital health platforms, many organizations struggle to analyze **user behavior**, **feature adoption**, and **platform usability** effectively.

This project bridges that gap by building a **Healthcare Data Analytics Platform** that not only **visualizes engagement trends** but also leverages **predictive modeling** to identify factors influencing user satisfaction, accessibility, and health-related interactions. The insights derived can serve as a **benchmark** for developing better healthcare technology solutions that cater to patient and caregiver needs.

## Dataset
We collaborated with the **Cancer Awareness and Rehabilitation Foundation (CARF NGO)** to obtain a dataset consisting of **200+ responses** from **cancer patients and their families**. The dataset was collected via surveys before and after the development of [CARF's website](https://carfngo.org/).

📌 This dataset provides a **real-world benchmark** for analyzing patient engagement, accessibility challenges, and personalized content preferences in healthcare platforms. The insights derived can help organizations optimize digital health tools and improve patient care experiences.

## Solution Overview
The **Healthcare Data Analytics Platform** enables **data-driven healthcare enhancement** through:

### 🔹 **Technical Features:**
- **📊 Advanced Data Visualization**: Interactive visual reports using **Power BI, Plotly, and Seaborn** to analyze patient trends and website engagement patterns.
- **📈 Predictive Analytics**: Linear, Logistic, and Multiple Regression models implemented using **Scikit-Learn** to forecast **user satisfaction, feature engagement, and accessibility factors**.
- **⚙️ ML-Powered Decision Support**: Machine learning models trained on survey data to identify key **drivers of patient engagement**.
- **🛠 Feature Benchmarking**: Analysis of **website performance (load time, navigation style, multilingual impact, and accessibility importance)** to **guide healthcare tech improvements**.
- **🌐 Real-Time Analytics Dashboard**: A **Streamlit-based** interactive dashboard for analyzing patient feedback and visualizing trends.

## Technologies Used
We integrated a **robust tech stack** to enable data processing, predictive modeling, and real-time analytics:

### **🧠 Machine Learning & Data Processing:**
- `pandas`, `numpy` → Data handling & preprocessing
- `scikit-learn` → Regression models (Linear, Logistic, Multiple Regression)
- `joblib` → Model serialization for efficient reuse

### **📊 Data Visualization & Reporting:**
- `Power BI` → Interactive dashboards & real-time insights
- `matplotlib`, `seaborn` → Exploratory Data Analysis (EDA)
- `plotly`, `dash` → Interactive charts & analytics

### **💻 Web & Dashboard Development:**
- `streamlit` → Web-based data analytics dashboard
- `FastAPI` (Future Scope) → API integration for real-time healthcare analytics
- `Flask` (Alternative) → Lightweight deployment option

### **☁️ Deployment & Cloud:**
- **Streamlit Cloud** → Hosting the interactive analytics platform
- **AWS RDS / MongoDB Atlas (Future Scope)** → Cloud-based database for real-time storage

## Machine Learning Models Implemented
We leveraged **Supervised Learning** techniques for predictive modeling:

### **1️⃣ Linear Regression**
- **User Experience Analysis** → Predicts user satisfaction based on **engagement and website performance metrics**.
- **Feature Importance** → Identifies key **health-tech tools that improve accessibility and engagement**.

### **2️⃣ Logistic Regression**
- **Cancer Impact Prediction** → Determines if a user has been affected by cancer based on their **interaction with healthcare resources**.
- **Symptom Checker Usage** → Predicts likelihood of using **self-diagnosis tools**.

### **3️⃣ Multiple Regression**
- **Clicks to Find Information** → Evaluates ease of navigation based on **website UI, mobile responsiveness, and design consistency**.
- **Personalized Content Impact** → Determines the effect of **customized health content** on user engagement.

## Deployment
🚀 **Live Demo**: [Healthcare Data Analytics Platform](https://capstone-project-deployed.streamlit.app/)

## Power BI Dashboards
📊 **Power BI Reports**: [Power BI Folder](#) *(Replace with actual link to Power BI folder in repo)*

## How to Run the Project Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Healthcare-Data-Analytics.git
   cd Healthcare-Data-Analytics
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Future Enhancements
- **🧠 AI-Powered Insights** → Implement **Deep Learning (TensorFlow/PyTorch)** for better **predictive healthcare analytics**.
- **📡 Real-Time API** → Integrate **FastAPI** to process **real-time patient feedback**.
- **💾 Cloud-Based Storage** → Store patient engagement data on **AWS RDS or MongoDB Atlas**.

## Contributing
We welcome contributions! Feel free to submit **issues**, **feature requests**, or **pull requests** to enhance the project.

## License
📜 This project is licensed under the MIT License.

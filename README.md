# Used-Car-price-prediction
This project focuses on predicting the resale price of used cars using Machine Learning techniques. The model is trained on historical car data and learns patterns based on various features such as brand, model, year, fuel type, transmission, kilometers driven, and owner type.
                              
                              #PROJECT STRUCTURE
D:\New folder\
│
├── app.py
│
├── data\
│     └── used_cars.csv
│
├── model\
│
└── src\
      ├── __init__.py
      ├── train.py
      ├── api.py
      └── data_preprocess.py

#Steps to run this project:

1) Start FastAPI API Server
   open cmd:
    cd "D:\New folder"
    python -m uvicorn api:app --reload --app-dir ".\src"

2) Launch Streamlit Website
  open cmd new terminal or in VS Code (you can launch from anywhere)
  
   cd "D:\New folder"
   python -m streamlit run app.py

✅ Your Website Will Open





3) If You Change ML Code / Dataset:
   Open cmd and run training again
      cd "D:\New folder"
      python -m src.train

4) Then restart FastAPI.

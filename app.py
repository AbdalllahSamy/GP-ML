from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# تحميل الملفات المحفوظة
model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
label_encoders = joblib.load(os.path.join(BASE_DIR, 'label_encoders.pkl'))
output_encoders = joblib.load(os.path.join(BASE_DIR, 'output_encoders.pkl'))
X_columns = joblib.load(os.path.join(BASE_DIR, 'X_columns.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # حساب BMI
    data['BMI'] = data['Weight'] / (data['Height'] ** 2)
    
    # ترميز القيم النصية
    for col in ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']:
        if col in data:
            # التأكد أن القيمة موجودة في ال encoder
            if data[col] not in label_encoders[col].classes_:
                return jsonify({"error": f"Value '{data[col]}' not recognized for '{col}'"}), 400
            data[col] = label_encoders[col].transform([data[col]])[0]
        else:
            return jsonify({"error": f"Missing field '{col}'"}), 400

    # تحويل البيانات إلى DataFrame بنفس ترتيب الأعمدة
    input_df = pd.DataFrame([data])[X_columns]

    # التنبؤ
    predictions = model.predict(input_df)

    outputs = ['Exercises', 'Equipment', 'Diet', 'Recommendation']
    result = {}

    for i, col in enumerate(outputs):
        class_index = predictions[i].argmax()
        class_name = output_encoders[col].inverse_transform([class_index])[0]
        result[col] = class_name

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

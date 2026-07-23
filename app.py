from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import io
import re
import platform
import shutil
import os

app = Flask(__name__)

# --- OCR setup ---
import pytesseract
from PIL import Image

if platform.system() == 'Windows' and shutil.which('tesseract') is None:
    default_windows_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(default_windows_path):
        pytesseract.pytesseract.tesseract_cmd = default_windows_path

# Load model, scaler, and imputer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# REMOVED 'dpf' from here so it doesn't appear in the UI or extraction
FIELDS = [
    'pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
    'insulin', 'bmi', 'age'
]

COLS_WITH_ZEROS = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']

# REMOVED 'dpf' aliases
HEADER_ALIASES = {
    'pregnancies': ['pregnancies', 'pregnancy', 'num_pregnancies'],
    'glucose': ['glucose', 'glucose (mg/dl)', 'glucose_mg_dl', 'blood glucose'],
    'bloodpressure': ['bloodpressure', 'blood pressure', 'bp', 'blood pressure (mm hg)', 'diastolic bp'],
    'skinthickness': ['skinthickness', 'skin thickness', 'skin fold', 'triceps skinfold (mm)'],
    'insulin': ['insulin', 'insulin (uu/ml)', 'serum insulin'],
    'bmi': ['bmi', 'body mass index'],
    'age': ['age', 'age (years)', 'patient age'],
}

def normalize_header(h):
    return str(h).strip().lower()

def matches_field(label, field):
    return any(alias in label for alias in HEADER_ALIASES[field])

def get_table_reader(file):
    filename = file.filename.lower()
    if filename.endswith('.csv'):
        content = file.stream.read().decode('utf-8')
        return lambda header: pd.read_csv(io.StringIO(content), header=header)
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        content = file.stream.read()
        return lambda header: pd.read_excel(io.BytesIO(content), sheet_name=0, header=header)
    raise ValueError('Unsupported file type.')

def extract_from_wide_table(df):
    df = df.copy()
    df.columns = [normalize_header(c) for c in df.columns]
    if df.empty: return {}
    row = df.iloc[0]
    result = {}
    for field in FIELDS:
        found_value = None
        matched_col = next((c for c in df.columns if matches_field(c, field)), None)
        if matched_col is not None:
            try:
                found_value = float(row[matched_col])
            except (ValueError, TypeError):
                found_value = None
        result[field] = found_value
    return result

def extract_from_key_value_pairs(df_raw):
    result = {}
    if df_raw.shape[1] < 2: return result
    for _, raw_row in df_raw.iterrows():
        label_cell = raw_row.iloc[0]
        value_cell = raw_row.iloc[1]
        if pd.isna(label_cell): continue
        label = normalize_header(label_cell)
        for field in FIELDS:
            if matches_field(label, field):
                try:
                    result[field] = float(value_cell)
                except (ValueError, TypeError):
                    result[field] = None
                break
    return result

def package_result(values):
    return {
        field: {
            "value": (None if field in COLS_WITH_ZEROS and values.get(field) == 0 else values.get(field)),
            "status": "found" if values.get(field) is not None else "not_found"
        }
        for field in FIELDS
    }

def extract_fields(file):
    read = get_table_reader(file)
    wide_result = extract_from_wide_table(read(header=0))
    kv_result = extract_from_key_value_pairs(read(header=None))
    
    wide_found = sum(1 for v in wide_result.values() if v is not None)
    kv_found = sum(1 for v in kv_result.values() if v is not None)
    
    best = wide_result if wide_found >= kv_found else kv_result
    return package_result(best)

def extract_fields_from_image(file):
    image = Image.open(file.stream)
    text = pytesseract.image_to_string(image).lower()
    values = {}
    for field in FIELDS:
        found = None
        for alias in HEADER_ALIASES[field]:
            pattern = re.escape(alias) + r'[^0-9\n]{0,15}?(\d+\.?\d*)'
            match = re.search(pattern, text)
            if match:
                try:
                    found = float(match.group(1))
                    break
                except ValueError: continue
        values[field] = found
    return package_result(values)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = file.filename.lower()
    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            extracted = extract_fields_from_image(file)
        else:
            extracted = extract_fields(file)
        
        not_found = [f for f, v in extracted.items() if v['status'] == 'not_found']
        return jsonify({
            'extracted': extracted,
            'message': 'Some fields missing — please fill manually.' if not_found else 'All fields found.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        user_values = [float(data[field]) for field in FIELDS]

        df = pd.DataFrame([user_values], columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'Age'
        ])

        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            if df.loc[0, col] == 0:
                df.loc[0, col] = None

        df[cols_with_zeros] = imputer.transform(df[cols_with_zeros])
        input_scaled = scaler.transform(df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'probability': round(float(probability), 3),
            'message': 'Elevated risk — consult a professional' if prediction == 1 else 'Low risk'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import io
import re
import platform
import shutil

app = Flask(__name__)

# --- OCR setup (Windows needs the Tesseract binary installed separately from pytesseract) ---
import pytesseract
from PIL import Image

if platform.system() == 'Windows' and shutil.which('tesseract') is None:
    # pytesseract is just a Python wrapper -- it calls out to the real Tesseract
    # executable, which on Windows is NOT installed by pip and must be installed
    # separately. This is the default install location for the UB-Mannheim build.
    default_windows_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    import os
    if os.path.exists(default_windows_path):
        pytesseract.pytesseract.tesseract_cmd = default_windows_path
    # If Tesseract isn't found here either, OCR calls will raise a clear error
    # later (caught in the /upload route) rather than failing silently.

# Load model, scaler, and imputer (all three were saved during training)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# Canonical field order the model expects
FIELDS = [
    'pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
    'insulin', 'bmi', 'dpf', 'age'
]

# Columns where 0 is an invalid placeholder for "missing", matching training-time logic
COLS_WITH_ZEROS = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']

# Maps many possible CSV header spellings to our canonical field names.
# Add more variants here as you encounter real-world export formats.
HEADER_ALIASES = {
    'pregnancies': ['pregnancies', 'pregnancy', 'num_pregnancies'],
    'glucose': ['glucose', 'glucose (mg/dl)', 'glucose_mg_dl', 'blood glucose'],
    'bloodpressure': ['bloodpressure', 'blood pressure', 'bp', 'blood pressure (mm hg)', 'diastolic bp'],
    'skinthickness': ['skinthickness', 'skin thickness', 'skin fold', 'triceps skinfold (mm)'],
    'insulin': ['insulin', 'insulin (uu/ml)', 'serum insulin'],
    'bmi': ['bmi', 'body mass index'],
    'dpf': ['dpf', 'diabetes pedigree function', 'pedigree'],
    'age': ['age', 'age (years)', 'patient age'],
}


def normalize_header(h):
    return str(h).strip().lower()


def matches_field(label, field):
    """
    Returns True if a (normalized) header/label text refers to the given
    canonical field. Uses substring matching rather than exact equality,
    since real-world headers often carry extra text the alias list won't
    contain verbatim -- e.g. "Skin Thickness (mm)" vs alias "skin thickness",
    or "Insulin (µU/mL)" vs alias "insulin" (note the µ symbol).
    """
    return any(alias in label for alias in HEADER_ALIASES[field])


def get_table_reader(file):
    """
    Reads the uploaded file's raw bytes/text ONCE into memory, then returns a
    function that can produce a fresh DataFrame from that same content as many
    times as needed (e.g. once with header=0, once with header=None) without
    re-reading the file stream, which can only be consumed once.
    """
    filename = file.filename.lower()

    if filename.endswith('.csv'):
        content = file.stream.read().decode('utf-8')
        def read(header):
            return pd.read_csv(io.StringIO(content), header=header)
        return read

    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        # pandas picks the right engine automatically (openpyxl for .xlsx,
        # xlrd for legacy .xls) as long as both are installed
        content = file.stream.read()
        def read(header):
            return pd.read_excel(io.BytesIO(content), sheet_name=0, header=header)
        return read

    raise ValueError('Unsupported file type. Please upload a .csv, .xlsx, or .xls file.')


def extract_from_wide_table(df):
    """
    Layout A: one header row of field names across the top, values in the row below.
    e.g.  Glucose | BMI | Age
            150   | 25  | 28
    Returns { field: value_or_None }.
    """
    df = df.copy()
    df.columns = [normalize_header(c) for c in df.columns]

    if df.empty:
        return {}

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
    """
    Layout B: each row is one field, label in column 0, value in column 1.
    e.g.  Glucose (mg/dL)   | 150
          Blood Pressure    | 70
    This is the layout your test1.xlsx file actually uses.
    Returns { field: value_or_None }.
    """
    result = {}
    if df_raw.shape[1] < 2:
        return result

    for _, raw_row in df_raw.iterrows():
        label_cell = raw_row.iloc[0]
        value_cell = raw_row.iloc[1]
        if pd.isna(label_cell):
            continue

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
    """
    Converts a plain { field: value_or_None } dict into the standard shape
    used everywhere in this app: { field: {"value": ..., "status": "found"|"not_found"} }.
    Used by BOTH the table-based (CSV/Excel) and OCR (image) extraction paths,
    so the frontend always sees the same structure regardless of source --
    this is also what drives the "highlight missing fields for manual entry"
    behavior automatically, with no extra logic needed per file type.
    """
    return {
        field: {
            "value": (None if field in COLS_WITH_ZEROS and values.get(field) == 0 else values.get(field)),
            "status": "found" if values.get(field) is not None else "not_found"
        }
        for field in FIELDS
    }


def extract_fields(file):
    """
    Reads a CSV/Excel file and tries BOTH supported layouts:
    1. Wide table (header row + one data row)
    2. Key-value pairs (label in column A, value in column B, one field per row)
    Whichever layout finds more fields wins; the two are not mixed.
    Returns a dict: { field: {"value": float|None, "status": "found"|"not_found"} }
    """
    read = get_table_reader(file)

    wide_result = extract_from_wide_table(read(header=0))
    wide_found = sum(1 for v in wide_result.values() if v is not None)

    kv_result = extract_from_key_value_pairs(read(header=None))
    kv_found = sum(1 for v in kv_result.values() if v is not None)

    best = wide_result if wide_found >= kv_found else kv_result
    return package_result(best)


def extract_fields_from_image(file):
    """
    Runs OCR on an uploaded image (e.g. a photographed/scanned blood test
    report) and pulls out whatever fields it can find using regex matching
    against the same alias list used for CSV/Excel headers.

    OCR text is noisy and layout varies wildly between lab providers, so this
    is intentionally permissive: for each field, it searches the WHOLE
    extracted text for any known alias, then captures the nearest following
    number (allowing up to ~15 characters of separator text/punctuation/units
    in between, e.g. "Glucose (mg/dL): 150" or "Glucose ... 150 mg/dL").

    Fields it cannot confidently find are left as None and come back tagged
    "not_found" -- exactly like the CSV/Excel path -- so the frontend review
    step naturally adjusts: found fields pre-fill and get flagged for
    verification, missing fields get flagged red for manual entry. No
    special-casing is needed for "incomplete" OCR results.
    """
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
                except ValueError:
                    continue
        values[field] = found

    return package_result(values)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Accepts a CSV, Excel, or image file, extracts whatever fields it can
    find, and returns them to the frontend for the user to review/edit
    BEFORE prediction. This route never calls the model directly --
    extraction and prediction are kept as separate steps on purpose.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = file.filename.lower()
    table_extensions = ('.csv', '.xlsx', '.xls')
    image_extensions = ('.png', '.jpg', '.jpeg')

    if not filename.endswith(table_extensions + image_extensions):
        return jsonify({'error': 'Only .csv, .xlsx, .xls, .png, .jpg, or .jpeg files are supported'}), 400

    try:
        if filename.endswith(image_extensions):
            extracted = extract_fields_from_image(file)
        else:
            extracted = extract_fields(file)

        not_found = [f for f, v in extracted.items() if v['status'] == 'not_found']

        return jsonify({
            'extracted': extracted,
            'message': 'Some fields could not be found in the file — please fill them in manually.'
                       if not_found else 'All fields were found. Please review before predicting.'
        })
    except Exception as e:
        return jsonify({'error': f'Could not parse file: {str(e)}'}), 400


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction entry point used by BOTH the manual form and the
    upload-review flow. Always runs the same impute -> scale -> predict
    pipeline used at training time.
    """
    try:
        data = request.get_json()
        features = [[float(data[field]) for field in FIELDS]]

        df = pd.DataFrame(features, columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])

        # Replace 0 with NaN in the same columns treated this way at training time
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            if df.loc[0, col] == 0:
                df.loc[0, col] = None

        df[cols_with_zeros] = imputer.transform(df[cols_with_zeros])

        input_scaled = scaler.transform(df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        result = ('Elevated risk — please consult a healthcare professional'
                  if prediction == 1 else 'Low risk')

        return jsonify({
            'prediction': int(prediction),
            'probability': round(float(probability), 3),
            'message': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

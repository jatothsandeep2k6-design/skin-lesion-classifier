# ============================================================
# predict.py
# PURPOSE: All prediction logic for the Streamlit app.
#          Auto-downloads model from Google Drive if needed.
# ============================================================

import os
import sys
import pickle
import numpy as np
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config

# ============================================================
# FUNCTION 1 — load_model_cached()
# ============================================================
@st.cache_resource
def load_model_cached():
    import tensorflow as tf

    # Google Drive file ID from your shared link
    DRIVE_FILE_ID = "1Zj_o3hrKR2ZViFNIkmA3a8UhsFNhXGH7"
    model_path   = config.MODEL_SAVE_PATH
    encoder_path = config.ENCODER_SAVE_PATH

    # Download model if not found locally
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, model_path, quiet=False)
            print("Model downloaded successfully!")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download model: {e}\n"
                f"Make sure Google Drive file is set to 'Anyone with link'."
            )

    # Load model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")

    # Load encoder
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Encoder not found at: {encoder_path}"
        )
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print("Encoder loaded successfully")

    return model, encoder


# ============================================================
# FUNCTION 2 — preprocess_image()
# ============================================================
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert('RGB')
    img = img.resize(config.IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ============================================================
# FUNCTION 3 — preprocess_metadata()
# ============================================================
def preprocess_metadata(age, sex, localization, encoder):
    age_normalized = float(age) / 100.0
    sex_mapping = {'Male': 0.0, 'Female': 1.0, 'Unknown': 0.5}
    sex_encoded = sex_mapping.get(sex, 0.5)
    try:
        known = list(encoder.classes_)
        loc_value = localization if localization in known else 'unknown'
        loc_encoded = float(encoder.transform([loc_value])[0])
    except Exception:
        loc_encoded = 0.0
    metadata_array = np.array(
        [[age_normalized, sex_encoded, loc_encoded]],
        dtype=np.float32
    )
    return metadata_array


# ============================================================
# FUNCTION 4 — predict()
# ============================================================
def predict(model, image_array, metadata_array):
    raw_predictions = model.predict(
        [image_array, metadata_array],
        verbose=0
    )
    probabilities    = raw_predictions[0]
    predicted_index  = int(np.argmax(probabilities))
    predicted_class  = config.CLASS_NAMES[predicted_index]
    predicted_label  = config.CLASS_FULL_NAMES.get(predicted_class, predicted_class)
    confidence       = float(probabilities[predicted_index]) * 100.0
    all_probabilities = {
        config.CLASS_FULL_NAMES.get(cls, cls): float(prob) * 100.0
        for cls, prob in zip(config.CLASS_NAMES, probabilities)
    }
    return {
        'predicted_class'  : predicted_class,
        'predicted_label'  : predicted_label,
        'confidence'       : confidence,
        'all_probabilities': all_probabilities
    }


# ============================================================
# FUNCTION 5 — get_risk_level()
# ============================================================
def get_risk_level(predicted_class):
    if predicted_class in ['mel', 'bcc']:
        return {
            'level'  : 'HIGH RISK',
            'color'  : '#ef4444',
            'bg'     : '#fef2f2',
            'border' : '#fecaca',
            'message': 'Please consult a dermatologist immediately.',
            'emoji'  : '🔴'
        }
    elif predicted_class in ['akiec']:
        return {
            'level'  : 'MODERATE RISK',
            'color'  : '#f97316',
            'bg'     : '#fff7ed',
            'border' : '#fed7aa',
            'message': 'Medical consultation is recommended.',
            'emoji'  : '🟠'
        }
    else:
        return {
            'level'  : 'LOWER RISK',
            'color'  : '#22c55e',
            'bg'     : '#f0fdf4',
            'border' : '#bbf7d0',
            'message': 'Monitor regularly and follow up if concerned.',
            'emoji'  : '🟢'
        }


# ============================================================
# FUNCTION 6 — get_disease_description()
# ============================================================
def get_disease_description(predicted_class):
    descriptions = {
        'mel': {
            'name'       : 'Melanoma',
            'description': (
                'Melanoma is the most dangerous form of skin cancer. '
                'It develops from melanocytes, the cells that give skin '
                'its colour. Early detection is critical — when caught '
                'at Stage 1, the 5-year survival rate is 98%.'
            ),
            'symptoms'   : (
                'Irregular borders, multiple colours (brown, black, red, '
                'white or blue), asymmetrical shape, diameter larger than '
                '6mm, evolving size or colour.'
            ),
            'action'     : (
                'Seek immediate dermatologist consultation. '
                'Do not delay — early treatment is highly effective.'
            )
        },
        'nv': {
            'name'       : 'Melanocytic Nevi',
            'description': (
                'Melanocytic nevi are commonly known as moles. They are '
                'benign growths of melanocytes and are extremely common. '
                'Most moles are completely harmless and require no treatment.'
            ),
            'symptoms'   : (
                'Round or oval shape, uniform brown colour, smooth '
                'borders, consistent size usually under 6mm.'
            ),
            'action'     : (
                'Regular self-examination recommended. '
                'See a dermatologist if you notice any changes.'
            )
        },
        'bcc': {
            'name'       : 'Basal Cell Carcinoma',
            'description': (
                'Basal cell carcinoma is the most common form of skin cancer. '
                'It rarely spreads but can cause local tissue damage if untreated.'
            ),
            'symptoms'   : (
                'Pearly or waxy bump, flat flesh-coloured lesion, '
                'bleeding or scabbing sore that heals and returns.'
            ),
            'action'     : (
                'Consult a dermatologist promptly. '
                'BCC is highly treatable when caught early.'
            )
        },
        'akiec': {
            'name'       : 'Actinic Keratoses',
            'description': (
                'Actinic keratoses are rough, scaly patches caused by '
                'years of sun exposure. They are considered precancerous.'
            ),
            'symptoms'   : (
                'Rough, dry, scaly patch of skin, itching or burning '
                'in the affected area, colour ranging from pink to red.'
            ),
            'action'     : (
                'Schedule a dermatologist visit. '
                'Treatment is straightforward and highly effective when early.'
            )
        },
        'bkl': {
            'name'       : 'Benign Keratosis',
            'description': (
                'Benign keratosis is a common non-cancerous skin growth. '
                'They are completely harmless and do not require treatment.'
            ),
            'symptoms'   : (
                'Waxy, scaly, slightly raised growth, colour ranging '
                'from white to tan, brown, or black.'
            ),
            'action'     : (
                'No treatment required. '
                'Monitor for any changes and consult a dermatologist if needed.'
            )
        },
        'df': {
            'name'       : 'Dermatofibroma',
            'description': (
                'Dermatofibromas are common benign skin growths. '
                'They are harmless fibrous nodules.'
            ),
            'symptoms'   : (
                'Small, hard bump that may be red, pink, or brownish, '
                'often appears on the legs, dimples inward when pinched.'
            ),
            'action'     : (
                'No treatment is necessary. '
                'Consult a dermatologist if it changes in appearance.'
            )
        },
        'vasc': {
            'name'       : 'Vascular Lesions',
            'description': (
                'Vascular lesions are abnormalities of blood vessels in skin. '
                'Most are benign and harmless.'
            ),
            'symptoms'   : (
                'Bright red or purple spots, flat or slightly raised, '
                'may bleed easily if traumatised.'
            ),
            'action'     : (
                'Generally no treatment needed. '
                'See a dermatologist if the lesion bleeds frequently.'
            )
        }
    }
    return descriptions.get(predicted_class, {
        'name'       : predicted_class,
        'description': 'Please consult a dermatologist for details.',
        'symptoms'   : 'Consult a medical professional.',
        'action'     : 'Please seek medical advice.'
    })


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    print("predict.py loaded successfully")
    print(f"Model path : {config.MODEL_SAVE_PATH}")
    print(f"Drive ID   : 1Zj_o3hrKR2ZViFNIkmA3a8UhsFNhXGH7")
    print("All functions ready!")
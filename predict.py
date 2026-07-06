# ============================================================
# predict.py
# PURPOSE: All prediction logic for the Streamlit app.
#          Loads model, preprocesses image and metadata,
#          runs prediction, returns results.
# LOCATION: SkinLesionClassifier/predict.py (ROOT folder)
# ============================================================

import os
import sys
import pickle
import numpy as np
import streamlit as st
from PIL import Image

# Add project root to path so we can import config
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config

# ============================================================
# FUNCTION 1 — load_model_cached()
# Loads model and encoder ONCE and caches them
# Without caching, model reloads every time user clicks
# predict — would take 30 seconds each click
# ============================================================
@st.cache_resource
def load_model_cached():
    """
    Loads the trained model and localization encoder.
    @st.cache_resource means this runs ONLY ONCE when
    app starts — result is stored in memory permanently.
    """
    import tensorflow as tf

    # Check if model file exists
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {config.MODEL_SAVE_PATH}\n"
            f"Please complete training on Google Colab first,\n"
            f"then copy best_model.h5 to the saved_models/ folder."
        )

    # Load the trained Keras model
    print(f"Loading model from: {config.MODEL_SAVE_PATH}")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    print("Model loaded successfully")

    # Check if encoder file exists
    if not os.path.exists(config.ENCODER_SAVE_PATH):
        raise FileNotFoundError(
            f"Encoder not found at: {config.ENCODER_SAVE_PATH}\n"
            f"Please copy localization_encoder.pkl from Colab."
        )

    # Load the localization label encoder
    with open(config.ENCODER_SAVE_PATH, 'rb') as f:
        encoder = pickle.load(f)
    print("Encoder loaded successfully")

    return model, encoder


# ============================================================
# FUNCTION 2 — preprocess_image()
# Takes uploaded image file and prepares it for the model
# ============================================================
def preprocess_image(uploaded_file):
    """
    Preprocesses an uploaded image for model input.

    Steps:
    1. Open with PIL
    2. Convert to RGB (handles RGBA and grayscale)
    3. Resize to 224x224
    4. Normalize pixels to 0-1
    5. Add batch dimension

    Returns numpy array of shape (1, 224, 224, 3)
    """
    # Open image from uploaded file bytes
    img = Image.open(uploaded_file)

    # Convert to RGB — ensures exactly 3 colour channels
    # Some images are RGBA (4 channels) or grayscale (1 channel)
    img = img.convert('RGB')

    # Resize to exactly 224x224 — EfficientNetB0 requirement
    img = img.resize(config.IMAGE_SIZE)

    # Convert PIL image to numpy array
    # Shape becomes (224, 224, 3)
    img_array = np.array(img, dtype=np.float32)

    # Normalize pixel values from 0-255 to 0.0-1.0
    # Neural networks work better with small numbers
    img_array = img_array / 255.0

    # Add batch dimension: (224,224,3) becomes (1,224,224,3)
    # Model expects a batch even for single image
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ============================================================
# FUNCTION 3 — preprocess_metadata()
# Encodes age, sex, localization the same way as training
# ============================================================
def preprocess_metadata(age, sex, localization, encoder):
    """
    Encodes patient metadata exactly as done during training.

    age          : integer (0-100) → divided by 100
    sex          : string ('Male'/'Female'/'Unknown') → 0/1/0.5
    localization : string ('back','face' etc) → integer via encoder
    encoder      : saved LabelEncoder from training

    Returns numpy array of shape (1, 3)
    """

    # --- ENCODE AGE ---
    # Normalize age to 0-1 range (same as training)
    age_normalized = float(age) / 100.0

    # --- ENCODE SEX ---
    # Same mapping as data_loader.py
    sex_mapping = {
        'Male'   : 0.0,
        'Female' : 1.0,
        'Unknown': 0.5
    }
    sex_encoded = sex_mapping.get(sex, 0.5)

    # --- ENCODE LOCALIZATION ---
    # Use the saved encoder from training
    try:
        # Check if value was seen during training
        known = list(encoder.classes_)
        loc_value = localization if localization in known else 'unknown'
        loc_encoded = float(encoder.transform([loc_value])[0])
    except Exception:
        # If anything goes wrong use 0
        loc_encoded = 0.0

    # Build metadata array shape (1, 3)
    metadata_array = np.array(
        [[age_normalized, sex_encoded, loc_encoded]],
        dtype=np.float32
    )

    return metadata_array


# ============================================================
# FUNCTION 4 — predict()
# Runs the model and returns prediction results
# ============================================================
def predict(model, image_array, metadata_array):
    """
    Runs model prediction and returns formatted results.

    Returns dictionary with:
    - predicted_class   : short name e.g. 'mel'
    - predicted_label   : full name e.g. 'Melanoma'
    - confidence        : percentage e.g. 87.3
    - all_probabilities : dict of all 7 class probabilities
    """

    # Run model prediction
    # Returns array of shape (1, 7) — 7 probabilities
    raw_predictions = model.predict(
        [image_array, metadata_array],
        verbose=0   # suppress progress bar in app
    )

    # Get probabilities for all 7 classes
    # raw_predictions[0] removes the batch dimension
    probabilities = raw_predictions[0]

    # Get predicted class index (index of highest probability)
    predicted_index = int(np.argmax(probabilities))

    # Get predicted class short name
    predicted_class = config.CLASS_NAMES[predicted_index]

    # Get predicted class full name
    predicted_label = config.CLASS_FULL_NAMES.get(
        predicted_class, predicted_class
    )

    # Get confidence as percentage
    confidence = float(probabilities[predicted_index]) * 100.0

    # Build dictionary of all class probabilities
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
# Returns risk level based on predicted disease class
# ============================================================
def get_risk_level(predicted_class):
    """
    Returns risk level dictionary based on predicted class.

    HIGH     : mel (Melanoma), bcc (Basal Cell Carcinoma)
    MODERATE : akiec (Actinic Keratoses)
    LOW      : nv, bkl, df, vasc
    """

    high_risk = ['mel', 'bcc']
    moderate_risk = ['akiec']

    if predicted_class in high_risk:
        return {
            'level'  : 'HIGH RISK',
            'color'  : '#ef4444',
            'bg'     : '#fef2f2',
            'border' : '#fecaca',
            'message': 'Please consult a dermatologist immediately.',
            'emoji'  : '🔴'
        }
    elif predicted_class in moderate_risk:
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
# Returns information about each disease type
# ============================================================
def get_disease_description(predicted_class):
    """
    Returns a dictionary with disease information for display.
    Used in the expandable disease info section of the app.
    """

    descriptions = {
        'mel': {
            'name'      : 'Melanoma',
            'description': (
                'Melanoma is the most dangerous form of skin cancer. '
                'It develops from melanocytes, the cells that give skin '
                'its colour. Early detection is critical — when caught '
                'at Stage 1, the 5-year survival rate is 98%.'
            ),
            'symptoms'  : (
                'Irregular borders, multiple colours (brown, black, red, '
                'white or blue), asymmetrical shape, diameter larger than '
                '6mm, evolving size or colour.'
            ),
            'action'    : (
                'Seek immediate dermatologist consultation. '
                'Do not delay — early treatment is highly effective.'
            )
        },
        'nv': {
            'name'      : 'Melanocytic Nevi',
            'description': (
                'Melanocytic nevi are commonly known as moles. They are '
                'benign growths of melanocytes and are extremely common '
                'in the general population. Most moles are completely '
                'harmless and require no treatment.'
            ),
            'symptoms'  : (
                'Round or oval shape, uniform brown colour, smooth '
                'borders, consistent size (usually under 6mm), '
                'symmetrical appearance.'
            ),
            'action'    : (
                'Regular self-examination recommended. '
                'See a dermatologist if you notice any changes in '
                'size, shape, or colour.'
            )
        },
        'bcc': {
            'name'      : 'Basal Cell Carcinoma',
            'description': (
                'Basal cell carcinoma is the most common form of skin '
                'cancer. It arises from basal cells in the deepest layer '
                'of the epidermis. It rarely spreads to other parts of '
                'the body but can cause local tissue damage if untreated.'
            ),
            'symptoms'  : (
                'Pearly or waxy bump, flat flesh-coloured lesion, '
                'bleeding or scabbing sore that heals and returns, '
                'pink growth with raised edges.'
            ),
            'action'    : (
                'Consult a dermatologist promptly. '
                'BCC is highly treatable when caught early. '
                'Several effective treatment options are available.'
            )
        },
        'akiec': {
            'name'      : 'Actinic Keratoses',
            'description': (
                'Actinic keratoses are rough, scaly patches caused by '
                'years of sun exposure. They are considered precancerous '
                'because a small percentage can develop into squamous '
                'cell carcinoma if left untreated.'
            ),
            'symptoms'  : (
                'Rough, dry, scaly patch of skin, flat to slightly '
                'raised patch, hard, warty surface, itching or burning '
                'in the affected area, colour ranging from pink to red.'
            ),
            'action'    : (
                'Schedule a dermatologist visit. '
                'Treatment is straightforward and highly effective '
                'when addressed early.'
            )
        },
        'bkl': {
            'name'      : 'Benign Keratosis',
            'description': (
                'Benign keratosis (seborrheic keratosis) is a common '
                'non-cancerous skin growth that often appears as people '
                'age. They are completely harmless and do not require '
                'treatment unless they cause discomfort.'
            ),
            'symptoms'  : (
                'Waxy, scaly, slightly raised growth, colour ranging '
                'from white to light tan, brown, or black, round or '
                'oval shape, "stuck on" appearance.'
            ),
            'action'    : (
                'No treatment required for benign keratosis. '
                'Monitor for any changes and consult a dermatologist '
                'if the lesion changes rapidly.'
            )
        },
        'df': {
            'name'      : 'Dermatofibroma',
            'description': (
                'Dermatofibromas are common benign skin growths that '
                'most often appear on the legs. They are harmless '
                'fibrous nodules that may develop after a minor injury '
                'such as an insect bite or splinter.'
            ),
            'symptoms'  : (
                'Small, hard bump that may be red, pink, or brownish, '
                'often appears on the legs, dimples inward when pinched, '
                'may be slightly itchy or tender.'
            ),
            'action'    : (
                'No treatment is necessary. '
                'Consult a dermatologist if it changes in appearance '
                'or causes significant discomfort.'
            )
        },
        'vasc': {
            'name'      : 'Vascular Lesions',
            'description': (
                'Vascular lesions are abnormalities of blood vessels '
                'in the skin. They include cherry angiomas, spider '
                'angiomas, and pyogenic granulomas. Most are benign '
                'and harmless.'
            ),
            'symptoms'  : (
                'Bright red or purple spots, flat or slightly raised, '
                'may bleed easily if traumatised, size ranges from '
                'pinpoint to several centimetres.'
            ),
            'action'    : (
                'Generally no treatment needed. '
                'See a dermatologist if the lesion bleeds frequently '
                'or if you are concerned about its appearance.'
            )
        }
    }

    # Return description for predicted class
    # If class not found, return generic description
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
    print(f"Model path    : {config.MODEL_SAVE_PATH}")
    print(f"Encoder path  : {config.ENCODER_SAVE_PATH}")
    print("All 6 functions ready:")
    print("  load_model_cached()")
    print("  preprocess_image()")
    print("  preprocess_metadata()")
    print("  predict()")
    print("  get_risk_level()")
    print("  get_disease_description()")
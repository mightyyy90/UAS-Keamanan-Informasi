from flask import Flask, request, render_template, send_from_directory
import numpy as np
import pywt
import cv2
import hashlib
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def rdwt2(image):
    coeffs = pywt.dwtn(image, 'haar')
    return coeffs

def irdwt2(coeffs):
    image = pywt.idwtn(coeffs, 'haar')
    return image

def embed_watermark(image, watermark, secret_key):
    key_hash = hashlib.sha256(secret_key.encode()).digest()[:4]
    key = np.frombuffer(key_hash, dtype=np.uint8)

    coeffs = rdwt2(image)
    cA, (cH, cV, cD) = coeffs['aa'], (coeffs['da'], coeffs['ad'], coeffs['dd'])
    watermark = cv2.resize(watermark, (cA.shape[1], cA.shape[0]))
    key_matrix = np.tile(key, (watermark.shape[0], watermark.shape[1] // len(key)))
    key_matrix = key_matrix.astype(np.float32)
    watermark = cv2.add(watermark, key_matrix, dtype=cv2.CV_32F)
    cH_w = cH + watermark * 0.01
    cV_w = cV + watermark * 0.01
    cD_w = cD + watermark * 0.01
    coeffs['da'], coeffs['ad'], coeffs['dd'] = cH_w, cV_w, cD_w
    watermarked_image = irdwt2(coeffs)
    return watermarked_image

def extract_watermark(image, original_image, secret_key):
    key_hash = hashlib.sha256(secret_key.encode()).digest()[:4]
    key = np.frombuffer(key_hash, dtype=np.uint8)
    coeffs_image = rdwt2(image)
    coeffs_original = rdwt2(original_image)
    cH_w, cV_w, cD_w = coeffs_image['da'], coeffs_image['ad'], coeffs_image['dd']
    cH_o, cV_o, cD_o = coeffs_original['da'], coeffs_original['ad'], coeffs_original['dd']
    watermark_H = (cH_w - cH_o) / 0.01
    watermark_V = (cV_w - cV_o) / 0.01
    watermark_D = (cD_w - cD_o) / 0.01
    watermark = (watermark_H + watermark_V + watermark_D) / 3
    key_matrix = np.tile(key, (watermark.shape[0], watermark.shape[1] // len(key)))
    key_matrix = key_matrix.astype(np.float32)
    watermark = cv2.subtract(watermark, key_matrix, dtype=cv2.CV_32F)
    return watermark

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    image_file = request.files['image']
    watermark_file = request.files['watermark']
    secret_key = request.form['secret_key']

    if image_file and watermark_file and secret_key:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], watermark_file.filename)
        image_file.save(image_path)
        watermark_file.save(watermark_path)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        watermarked_image = embed_watermark(image, watermark, secret_key)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'watermarked_image.png')
        cv2.imwrite(result_path, watermarked_image)

        return render_template('result.html', message='Watermarked Image', image_url=result_path)
    return 'Failed to embed watermark', 400

@app.route('/extract', methods=['POST'])
def extract():
    watermarked_image_file = request.files['watermarked_image']
    original_image_file = request.files['original_image']
    secret_key = request.form['secret_key']

    if watermarked_image_file and original_image_file and secret_key:
        watermarked_image_path = os.path.join(app.config['UPLOAD_FOLDER'], watermarked_image_file.filename)
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_image_file.filename)
        watermarked_image_file.save(watermarked_image_path)
        original_image_file.save(original_image_path)

        watermarked_image = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        extracted_watermark = extract_watermark(watermarked_image, original_image, secret_key)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'extracted_watermark.png')
        cv2.imwrite(result_path, extracted_watermark)

        return render_template('result.html', message='Extracted Watermark', image_url=result_path)
    return 'Failed to extract watermark', 400

if __name__ == '__main__':
    app.run(debug=True)

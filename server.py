
from flask import Flask, request, send_file
import cv2
import numpy as np
import tempfile

app = Flask(__name__)

@app.route('/remove-watermark', methods=['POST'])
def remove_watermark():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = image.shape[:2]
    cv2.rectangle(mask, (50, h - 100), (w - 50, h - 20), 255, -1)

    cleaned = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, cleaned)
    return send_file(temp_file.name, mimetype='image/jpeg')

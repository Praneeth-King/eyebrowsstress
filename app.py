from flask import Flask, Response, render_template, jsonify
from test2 import VideoCamera
import os
import logging
import traceback
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

def gen(test):
    """Video streaming generator function."""
    try:
        while True:
            frame = test.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        logger.error(f"Error in video streaming: {str(e)}")
        logger.error(traceback.format_exc())
        yield b''
       
@app.route("/predict", methods=['POST', 'GET'])
def predict():
    try:
        # First check if the required model files exist
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        emotion_model_path = "_mini_XCEPTION.102-0.66.hdf5"
        
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Missing required file: {predictor_path}")
        if not os.path.exists(emotion_model_path):
            raise FileNotFoundError(f"Missing required file: {emotion_model_path}")
            
        camera = VideoCamera()
        return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    # Check if templates directory exists
    if not os.path.exists('templates'):
        logger.error("Templates directory not found!")
        sys.exit(1)
        
    # Check if index.html exists
    if not os.path.exists('templates/index.html'):
        logger.error("index.html not found in templates directory!")
        sys.exit(1)
        
    app.run(debug=True, use_reloader=False)
"""
AstroIdentify Web Application
A Flask web app for constellation identification using deep learning.
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import logging
from datetime import datetime

# Import our constellation classifier
from constellation_classifier import ConstellationClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the constellation classifier
try:
    classifier = ConstellationClassifier()
    logger.info("Constellation classifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize classifier: {str(e)}")
    classifier = None


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    """Convert image to base64 string for display in HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None


@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction."""
    if classifier is None:
        return jsonify({'error': 'Classifier not available'}), 500
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the uploaded file
            file.save(filepath)
            logger.info(f"File saved: {filepath}")
            
            # Make prediction
            result = classifier.predict(filepath, return_probabilities=False, top_k=5)
            
            # Convert image to base64 for display
            image_base64 = image_to_base64(filepath)
            
            # Prepare response data
            response_data = {
                'success': True,
                'filename': filename,
                'image_data': image_base64,
                'prediction': result,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Clean up - remove uploaded file after processing
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {str(e)}")
            
            return render_template('result.html', data=response_data)
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)')
        return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for constellation prediction."""
    if classifier is None:
        return jsonify({'error': 'Classifier not available'}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        # Read image directly from memory without saving to disk
        image_bytes = io.BytesIO(file.read())
        image = Image.open(image_bytes)
        
        # Make prediction
        result = classifier.predict(image, return_probabilities=True, top_k=5)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info')
def api_model_info():
    """API endpoint to get model information."""
    if classifier is None:
        return jsonify({'error': 'Classifier not available'}), 500
    
    try:
        info = classifier.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page with information about the constellation classifier."""
    model_info = None
    if classifier:
        try:
            model_info = classifier.get_model_info()
        except Exception as e:
            logger.error(f"Error getting model info for about page: {str(e)}")
    
    return render_template('about.html', model_info=model_info)


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash("File is too large. Maximum size is 16MB.")
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
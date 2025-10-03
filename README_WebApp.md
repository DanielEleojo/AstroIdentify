# 🌟 AstroIdentify - Constellation Classification Web App

**AstroIdentify** is an AI-powered web application that identifies constellations from astronomical images using deep learning. Built with a ResNet18 neural network trained on constellation datasets, it can accurately classify 12 different constellations with confidence scores.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🤖 **AI-Powered Classification**: Advanced ResNet18 deep learning model
- 🎯 **High Accuracy**: Trained specifically for constellation recognition
- ⚡ **Fast Processing**: Get results in seconds
- 🌐 **Web Interface**: Beautiful, responsive web application
- 📱 **Mobile Friendly**: Works on all devices
- 🔒 **Secure**: Safe file handling and processing
- 📊 **Detailed Results**: Confidence scores and top-5 predictions
- 📚 **Educational**: Learn about each constellation

## 🌌 Supported Constellations

The model can identify the following 12 constellations:

- ♒ **Aquarius** (The Water Bearer)
- ♈ **Aries** (The Ram)
- ♋ **Cancer** (The Crab)
- ♑ **Capricornus** (The Sea Goat)
- ♊ **Gemini** (The Twins)
- ♌ **Leo** (The Lion)
- ♎ **Libra** (The Scales)
- ♓ **Pisces** (The Fishes)
- ♐ **Sagittarius** (The Archer)
- ♏ **Scorpius** (The Scorpion)
- ♉ **Taurus** (The Bull)
- ♍ **Virgo** (The Maiden)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Trained model files: `constellation_best_model.pt` and `constellation_classes.json`

### Installation

1. **Clone or download the project**
   ```bash
   cd AstroIdentify
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**
   - `constellation_best_model.pt` (trained model weights)
   - `constellation_classes.json` (class names)

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
AstroIdentify/
├── app.py                          # Flask web application
├── constellation_classifier.py      # Refactored model class
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── constellation_best_model.pt     # Trained model weights
├── constellation_classes.json      # Class names
├── templates/                      # HTML templates
│   ├── base.html                  # Base template
│   ├── index.html                 # Home page
│   ├── result.html                # Results page
│   ├── about.html                 # About page
│   ├── 404.html                   # 404 error page
│   └── 500.html                   # 500 error page
├── static/                        # Static assets
│   ├── css/
│   │   └── style.css             # Custom styles
│   └── js/
│       └── main.js               # JavaScript functionality
└── uploads/                       # Temporary upload directory
```

## 🔧 Configuration

### Environment Variables

You can customize the application by setting these environment variables:

```bash
# Flask configuration
FLASK_ENV=production          # or 'development'
FLASK_DEBUG=False            # or True for development
SECRET_KEY=your-secret-key   # Change this in production

# File upload settings
MAX_CONTENT_LENGTH=16777216  # 16MB in bytes
UPLOAD_FOLDER=uploads        # Upload directory
```

### Model Configuration

The `ConstellationClassifier` class accepts these parameters:

```python
classifier = ConstellationClassifier(
    model_path="constellation_best_model.pt",
    classes_path="constellation_classes.json"
)
```

## 🖥️ API Endpoints

### Web Interface
- `GET /` - Home page with upload form
- `POST /upload` - Process uploaded image and show results
- `GET /about` - About page with technical details

### REST API
- `POST /api/predict` - JSON API for predictions
- `GET /api/model-info` - Get model information

### Example API Usage

```python
import requests

# Upload and predict
with open('constellation_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict', 
                           files={'file': f})
    result = response.json()
    print(f"Predicted: {result['prediction']['predicted_constellation']}")
```

## 📊 Usage Examples

### Web Interface
1. Open `http://localhost:5000` in your browser
2. Upload a constellation image (PNG, JPG, JPEG, GIF, BMP, TIFF)
3. Get instant classification results with confidence scores
4. Learn about the identified constellation

### Programmatic Usage

```python
from constellation_classifier import ConstellationClassifier

# Initialize classifier
classifier = ConstellationClassifier()

# Predict from file path
result = classifier.predict('path/to/constellation_image.jpg')
print(f"Constellation: {result['predicted_constellation']}")
print(f"Confidence: {result['confidence_percent']}")

# Predict from PIL Image
from PIL import Image
image = Image.open('constellation_image.jpg')
result = classifier.predict(image)

# Get detailed results
result = classifier.predict(image, return_probabilities=True, top_k=5)
for pred in result['top_predictions']:
    print(f"{pred['constellation']}: {pred['confidence_percent']}")
```

## 🎨 Customization

### Styling
- Modify `static/css/style.css` for custom styling
- Update `templates/base.html` for layout changes
- Add custom JavaScript in `static/js/main.js`

### Model
- Replace `constellation_best_model.pt` with your trained model
- Update `constellation_classes.json` with your class names
- Modify `ConstellationClassifier` class for different architectures

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   FileNotFoundError: Model weights file not found
   ```
   - Ensure `constellation_best_model.pt` is in the project directory
   - Check the file path in `ConstellationClassifier`

2. **CUDA/GPU issues**
   ```
   RuntimeError: CUDA error or GPU not available
   ```
   - Install CPU-only PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
   - The model will automatically use CPU if GPU is unavailable

3. **Import errors**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   - Install requirements: `pip install -r requirements.txt`
   - Activate your virtual environment

4. **File upload issues**
   - Check file size (max 16MB)
   - Ensure supported image format
   - Verify upload directory permissions

### Development Mode

Run in development mode for debugging:
```bash
export FLASK_ENV=development  # On Windows: set FLASK_ENV=development
export FLASK_DEBUG=True       # On Windows: set FLASK_DEBUG=True
python app.py
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add some amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution
- Additional constellation support
- Improved model accuracy
- Enhanced UI/UX design
- Mobile app development
- API improvements
- Documentation updates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch** team for the deep learning framework
- **Flask** team for the web framework
- **Bootstrap** for the responsive UI components
- **Font Awesome** for the beautiful icons
- **Astronomy community** for constellation datasets and knowledge

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing issues in the repository
3. Create a new issue with detailed information
4. Contact the development team

---

**Happy constellation hunting! 🌟✨**
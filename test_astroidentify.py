"""
Test script to demonstrate AstroIdentify functionality
"""

import os
import sys

def test_constellation_classifier():
    """Test the constellation classifier functionality"""
    print("=" * 60)
    print("🌟 ASTROIDENTIFY - CONSTELLATION CLASSIFIER TEST")
    print("=" * 60)
    
    try:
        from constellation_classifier import ConstellationClassifier
        
        # Initialize the classifier
        print("Initializing constellation classifier...")
        classifier = ConstellationClassifier()
        
        # Print model information
        info = classifier.get_model_info()
        print("\n📊 MODEL INFORMATION:")
        print(f"   Architecture: {info['model_architecture']}")
        print(f"   Classes: {info['num_classes']}")
        print(f"   Device: {info['device']}")
        print(f"   Input Size: {info['input_size']}")
        
        print(f"\n🎯 SUPPORTED CONSTELLATIONS:")
        for i, constellation in enumerate(info['class_names'], 1):
            print(f"   {i:2d}. {constellation}")
        
        # Test with sample image if available
        test_images = [
            "denoise_testing/thresholded_denoised_aquarius_constellation_2.jpg",
            "denoise_testing/thresholded_denoised_leo_constellation.jpg",
            "denoise_testing/thresholded_denoised_virgo_constellation.jpg"
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"\n🔍 TESTING WITH: {img_path}")
                result = classifier.predict(img_path, top_k=3)
                
                print(f"   🏆 PREDICTION: {result['predicted_constellation']}")
                print(f"   💯 CONFIDENCE: {result['confidence_percent']}")
                print(f"   📈 TOP 3 PREDICTIONS:")
                
                for i, pred in enumerate(result['top_predictions'], 1):
                    print(f"      {i}. {pred['constellation']}: {pred['confidence_percent']}")
                break
        else:
            print(f"\n⚠️  No test images found. Place constellation images in:")
            for img_path in test_images:
                print(f"      - {img_path}")
        
        print(f"\n✅ CLASSIFIER TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_application():
    """Test web application components"""
    print("\n" + "=" * 60)
    print("🌐 WEB APPLICATION COMPONENT TEST")
    print("=" * 60)
    
    try:
        # Test Flask import
        import flask
        print(f"✅ Flask {flask.__version__} is available")
        
        # Test other dependencies
        from werkzeug.utils import secure_filename
        print("✅ Werkzeug is available")
        
        from PIL import Image
        print("✅ PIL/Pillow is available")
        
        # Check if all files exist
        required_files = [
            "app.py",
            "constellation_classifier.py",
            "constellation_best_model.pt",
            "constellation_classes.json",
            "templates/index.html",
            "templates/result.html",
            "templates/about.html",
            "static/css/style.css",
            "static/js/main.js"
        ]
        
        print(f"\n📁 CHECKING REQUIRED FILES:")
        all_files_exist = True
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} (MISSING)")
                all_files_exist = False
        
        if all_files_exist:
            print(f"\n🎉 ALL REQUIRED FILES ARE PRESENT!")
            print(f"\n🚀 TO START THE WEB APPLICATION:")
            print(f"   1. Open a terminal in this directory")
            print(f"   2. Run: python app.py")
            print(f"   3. Open your browser to: http://localhost:5000")
            print(f"   4. Upload a constellation image and get instant results!")
        else:
            print(f"\n⚠️  Some required files are missing!")
        
        return all_files_exist
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_instructions():
    """Show detailed usage instructions"""
    print("\n" + "=" * 60)
    print("📚 USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("""
🌟 ASTROIDENTIFY WEB APPLICATION

WHAT IT DOES:
   • Identifies constellations from astronomical images
   • Uses ResNet18 deep learning model
   • Supports 12 different constellations
   • Provides confidence scores and detailed results

HOW TO USE:

   METHOD 1: Web Interface (Recommended)
   =====================================
   1. Run: python app.py
   2. Open browser: http://localhost:5000
   3. Upload constellation image
   4. Get instant results!

   METHOD 2: Python API
   ====================
   from constellation_classifier import ConstellationClassifier
   
   classifier = ConstellationClassifier()
   result = classifier.predict('path/to/image.jpg')
   print(f"Constellation: {result['predicted_constellation']}")

SUPPORTED FORMATS:
   • PNG, JPG, JPEG, GIF, BMP, TIFF
   • Maximum file size: 16MB
   • Best results with clear star patterns

FEATURES:
   • 🤖 AI-powered classification
   • ⚡ Fast processing (seconds)
   • 📊 Confidence scores
   • 📱 Mobile-friendly interface
   • 🎯 Top-5 predictions
   • 📚 Educational constellation info

SUPPORTED CONSTELLATIONS:
   Aquarius • Aries • Cancer • Capricornus • Gemini • Leo
   Libra • Pisces • Sagittarius • Scorpius • Taurus • Virgo
""")

def main():
    """Main test function"""
    print("🌌 Welcome to AstroIdentify - Constellation Classification System")
    
    # Test the constellation classifier
    classifier_ok = test_constellation_classifier()
    
    # Test web application components
    webapp_ok = test_web_application()
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"   Constellation Classifier: {'✅ WORKING' if classifier_ok else '❌ FAILED'}")
    print(f"   Web Application Setup:    {'✅ READY' if webapp_ok else '❌ INCOMPLETE'}")
    
    if classifier_ok and webapp_ok:
        print(f"\n🎉 ASTROIDENTIFY IS READY TO USE!")
        print(f"   Run 'python app.py' to start the web application")
    else:
        print(f"\n⚠️  Please fix the issues above before running the web app")
    
    return classifier_ok and webapp_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
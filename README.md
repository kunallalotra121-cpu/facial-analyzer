# 🔍 FaceAnalyzer Pro - Advanced Facial Analysis Platform

A comprehensive Python Flask web application that provides detailed facial analysis using computer vision and machine learning techniques. Upload a photo and get accurate ratings and insights about facial features, symmetry, proportions, and overall attractiveness.

## 🌟 Features

### 📊 Comprehensive Analysis
- **Facial Symmetry**: Advanced pixel-level symmetry detection and scoring
- **Golden Ratio Analysis**: Proportional harmony measurement based on classical beauty standards
- **Skin Quality Assessment**: Texture analysis and uniformity evaluation
- **Eye Analysis**: Spacing, size, symmetry, and positioning evaluation
- **Feature Detection**: Nose, mouth, and jawline characteristic analysis
- **Overall Rating**: 100-point scale with detailed explanations

### 🎨 Modern Web Interface
- **Multi-page Design**: Home, Upload, Results, About, and Contact pages
- **Responsive Layout**: Mobile-friendly design with modern gradients
- **Interactive Elements**: Drag-and-drop upload, progress bars, animations
- **Real-time Processing**: Instant analysis with loading states
- **Error Handling**: User-friendly error messages and validation

### 🔧 Technical Capabilities
- **Computer Vision**: OpenCV-powered facial detection and analysis
- **Image Processing**: PIL/Pillow for advanced image manipulation
- **RESTful API**: JSON endpoints for programmatic access
- **File Management**: Secure upload handling with validation
- **Session Storage**: Temporary result caching for user sessions

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download the application files**
   ```bash
   # All files are located in /home/user/output/
   cd /home/user/output/
   ```

2. **Install required packages**
   ```bash
   pip install flask opencv-python pillow numpy werkzeug
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - Start analyzing faces!

## 📁 Project Structure

```
/home/user/output/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   ├── index.html        # Home page
│   ├── upload.html       # Upload interface
│   ├── results.html      # Analysis results
│   ├── about.html        # About page
│   ├── contact.html      # Contact page
│   ├── 404.html          # Error page
│   └── 500.html          # Server error page
├── uploads/              # Temporary image storage
└── README.md             # This file
```

## 🔍 How It Works

### 1. Image Upload
- Drag and drop or click to select image files
- Supports PNG, JPG, JPEG, GIF formats (max 16MB)
- Real-time file validation and preview

### 2. Facial Detection
- OpenCV Haar Cascade classifiers detect faces and eyes
- Extracts facial regions for detailed analysis
- Handles multiple faces (analyzes the largest/primary face)

### 3. Analysis Process
- **Symmetry Calculation**: Compares left and right facial halves
- **Golden Ratio Measurement**: Evaluates proportional harmony
- **Skin Quality Assessment**: Analyzes texture and uniformity
- **Feature Analysis**: Detailed evaluation of eyes, nose, mouth, jawline
- **Scoring Algorithm**: Weighted combination of all metrics

### 4. Results Display
- Interactive score visualization with animated progress bars
- Detailed breakdown of each analysis component
- Explanations and improvement suggestions
- Download and sharing capabilities

## 🎯 API Endpoints

### Core Endpoints
- `GET /` - Home page
- `GET /upload` - Upload interface
- `GET /results` - Analysis results page
- `GET /about` - About page
- `GET /contact` - Contact page

### API Endpoints
- `POST /api/upload` - File upload and analysis
- `POST /api/analyze` - Base64 image analysis
- `GET /api/health` - Health check

### Example API Usage

```python
import requests

# Upload and analyze image
with open('photo.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/upload', files=files)
    result = response.json()

print(f"Overall Rating: {result['overall_rating']}/100")
print(f"Symmetry Score: {result['symmetry_score']}/100")
```

## 📊 Analysis Metrics

### Scoring System (0-100 scale)
- **90-100**: Exceptional
- **80-89**: Very Good
- **70-79**: Good
- **60-69**: Average
- **Below 60**: Needs Improvement

### Analysis Components
1. **Facial Symmetry (25% weight)**
   - Left-right facial comparison
   - Pixel-level difference analysis

2. **Golden Ratio (15% weight)**
   - Proportional harmony measurement
   - Classical beauty standard compliance

3. **Skin Quality (20% weight)**
   - Texture smoothness analysis
   - Color uniformity evaluation

4. **Eye Analysis (20% weight)**
   - Spacing and positioning
   - Size consistency and symmetry

5. **Facial Harmony (20% weight)**
   - Nose, mouth, jawline evaluation
   - Overall feature coordination

## 🔒 Privacy & Security

- **No Permanent Storage**: Images are processed temporarily and deleted
- **Local Processing**: All analysis happens on your server
- **No Data Sharing**: Your images are never shared with third parties
- **Session-based**: Results are stored temporarily for your session only

## 🛠️ Customization

### Modifying Analysis Parameters
Edit the `FacialAnalyzer` class in `app.py`:

```python
# Adjust scoring weights
weights = {
    "symmetry": 0.25,        # Facial symmetry importance
    "golden_ratio": 0.15,    # Golden ratio importance
    "skin_quality": 0.20,    # Skin quality importance
    "eyes": 0.20,           # Eye analysis importance
    "facial_harmony": 0.20   # Overall harmony importance
}
```

### Adding New Features
1. Extend the `analyze_facial_features()` method
2. Update the scoring algorithm in `calculate_overall_rating()`
3. Modify templates to display new metrics

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **"No face detected"**
   - Ensure the image shows a clear, front-facing face
   - Check lighting and image quality
   - Avoid sunglasses or face coverings

3. **File upload errors**
   - Check file size (max 16MB)
   - Verify file format (PNG, JPG, JPEG, GIF)
   - Ensure proper file permissions

4. **Port already in use**
   ```bash
   # Change port in app.py
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

## 📈 Performance Optimization

### For Better Performance
- Use smaller image files (under 2MB recommended)
- Ensure good lighting in photos
- Use front-facing, centered photos
- Close other applications to free up memory

### Production Deployment
- Set `debug=False` in app.py
- Use a production WSGI server (Gunicorn, uWSGI)
- Configure proper logging
- Set up SSL/HTTPS
- Use a reverse proxy (Nginx)

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include error handling
- Write unit tests for new features

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For support, bug reports, or feature requests:
- Create an issue in the repository
- Contact: support@faceanalyzer.pro
- Documentation: Check the `/about` page in the application

## 🔄 Version History

### v1.0.0 (Current)
- Initial release with full facial analysis
- Multi-page web interface
- RESTful API endpoints
- Comprehensive scoring system
- Mobile-responsive design

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Flask framework for web development
- NumPy for numerical computations
- PIL/Pillow for image processing
- Beauty research and golden ratio studies

---

**FaceAnalyzer Pro** - Empowering self-understanding through advanced facial analysis technology.

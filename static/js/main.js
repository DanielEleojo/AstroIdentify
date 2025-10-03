// AstroIdentify JavaScript Functions

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Initialize file upload functionality
    initializeFileUpload();
    
    // Add loading states to buttons
    initializeButtonLoading();
    
    // Initialize constellation info tooltips
    initializeConstellationTooltips();
}

function initializeFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadArea || !fileInput) return;
    
    // Drag and drop handlers
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change handler
    fileInput.addEventListener('change', handleFileSelect);
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const fileInput = document.getElementById('fileInput');
        fileInput.files = files;
        handleFileSelect();
    }
}

function handleFileSelect() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    // Validate file type
    if (!isValidImageFile(file)) {
        showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, TIFF)', 'warning');
        clearFile();
        return;
    }
    
    // Validate file size (16MB limit)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size must be less than 16MB', 'warning');
        clearFile();
        return;
    }
    
    updateFileInfo(file);
    showImagePreview(file);
    enableSubmitButton();
}

function isValidImageFile(file) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff'];
    return allowedTypes.includes(file.type);
}

function updateFileInfo(file) {
    const fileName = document.getElementById('fileName');
    const fileInfo = document.getElementById('fileInfo');
    
    if (fileName && fileInfo) {
        fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
        fileInfo.style.display = 'block';
    }
}

function showImagePreview(file) {
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    
    if (!imagePreview || !previewImg) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        
        // Add fade-in animation
        imagePreview.style.opacity = '0';
        setTimeout(() => {
            imagePreview.style.transition = 'opacity 0.3s ease';
            imagePreview.style.opacity = '1';
        }, 10);
    };
    reader.readAsDataURL(file);
}

function enableSubmitButton() {
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.classList.remove('btn-outline-primary');
        submitBtn.classList.add('btn-primary');
    }
}

function clearFile() {
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const imagePreview = document.getElementById('imagePreview');
    const submitBtn = document.getElementById('submitBtn');
    
    if (fileInput) fileInput.value = '';
    if (fileInfo) fileInfo.style.display = 'none';
    if (imagePreview) imagePreview.style.display = 'none';
    
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.classList.remove('btn-primary');
        submitBtn.classList.add('btn-outline-primary');
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function initializeButtonLoading() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                const loadingText = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                
                submitBtn.innerHTML = loadingText;
                submitBtn.disabled = true;
                
                // Reset button if form submission fails
                setTimeout(() => {
                    if (submitBtn.innerHTML === loadingText) {
                        submitBtn.innerHTML = originalText;
                        submitBtn.disabled = false;
                    }
                }, 10000); // 10 second timeout
            }
        });
    });
}

function initializeConstellationTooltips() {
    const constellationInfo = {
        'Aquarius': 'The Water Bearer - Best visible in autumn',
        'Aries': 'The Ram - Best visible in autumn and early winter',
        'Cancer': 'The Crab - Best visible in late winter and spring',
        'Capricornus': 'The Sea Goat - Best visible in late summer and early autumn',
        'Gemini': 'The Twins - Best visible in winter and spring',
        'Leo': 'The Lion - Best visible in spring',
        'Libra': 'The Scales - Best visible in late spring and early summer',
        'Pisces': 'The Fishes - Best visible in autumn',
        'Sagittarius': 'The Archer - Best visible in summer',
        'Scorpius': 'The Scorpion - Best visible in summer',
        'Taurus': 'The Bull - Best visible in winter',
        'Virgo': 'The Maiden - Best visible in spring and early summer'
    };
    
    // Add tooltips to constellation names
    document.querySelectorAll('[data-constellation]').forEach(element => {
        const constellation = element.getAttribute('data-constellation');
        if (constellation && constellationInfo[constellation]) {
            element.setAttribute('title', constellationInfo[constellation]);
            element.setAttribute('data-bs-toggle', 'tooltip');
        }
    });
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.setAttribute('role', 'alert');
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Insert at top of main content
    const main = document.querySelector('main');
    if (main) {
        main.insertBefore(alertDiv, main.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// API Functions for programmatic access
const AstroAPI = {
    async predict(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    },
    
    async getModelInfo() {
        try {
            const response = await fetch('/api/model-info');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Model info error:', error);
            throw error;
        }
    }
};

// Utility Functions
function animateProgressBar(progressBar, targetWidth, duration = 1000) {
    const startWidth = 0;
    const startTime = performance.now();
    
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentWidth = startWidth + (targetWidth - startWidth) * progress;
        progressBar.style.width = currentWidth + '%';
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    requestAnimationFrame(animate);
}

function copyToClipboard(text) {
    if (navigator.clipboard) {
        return navigator.clipboard.writeText(text);
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        return new Promise((resolve, reject) => {
            if (document.execCommand('copy')) {
                resolve();
            } else {
                reject(new Error('Copy failed'));
            }
            document.body.removeChild(textArea);
        });
    }
}

// Error Handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    showAlert('An unexpected error occurred. Please refresh the page and try again.', 'danger');
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showAlert('A network or processing error occurred. Please try again.', 'warning');
});

// Performance Monitoring
if ('performance' in window) {
    window.addEventListener('load', function() {
        setTimeout(() => {
            const perfData = performance.getEntriesByType('navigation')[0];
            if (perfData) {
                console.log(`Page load time: ${perfData.loadEventEnd - perfData.loadEventStart}ms`);
            }
        }, 0);
    });
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AstroAPI, formatFileSize, animateProgressBar, copyToClipboard };
}
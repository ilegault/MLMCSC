/**
 * Frontend JavaScript for MLMCSC Human-in-the-Loop Interface
 * 
 * Handles image upload, prediction display, label submission, and history viewing.
 */

class MLMCSCInterface {
    constructor() {
        this.currentPrediction = null;
        this.currentImageData = null;
        this.technicianId = this.getTechnicianId();
        
        // Live microscope properties
        this.cameraActive = false;
        this.autoPredictionEnabled = false;
        this.autoPredictionInterval = null;
        this.predictionIntervalSeconds = 3;
        
        this.initializeEventListeners();
        this.initializeLiveMicroscope();
        this.loadMetrics();
        this.loadHistory();
        this.checkCameraStatus();
        
        // Auto-refresh metrics and history every 30 seconds
        setInterval(() => {
            this.loadMetrics();
            this.loadHistory();
        }, 30000);
        
        // Check camera status every 10 seconds
        setInterval(() => {
            this.checkCameraStatus();
        }, 10000);
    }
    
    getTechnicianId() {
        // Get or create technician ID
        let techId = localStorage.getItem('technician_id');
        if (!techId) {
            techId = prompt('Please enter your technician ID:') || 'anonymous';
            localStorage.setItem('technician_id', techId);
        }
        return techId;
    }
    
    initializeEventListeners() {
        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');
        
        uploadArea.addEventListener('click', () => imageInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = '#f0f0f0';
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.backgroundColor = '';
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = '';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleImageUpload(files[0]);
            }
        });
        
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageUpload(e.target.files[0]);
            }
        });
        
        // Slider handling
        const shearSlider = document.getElementById('shearSlider');
        const shearValue = document.getElementById('shearValue');
        
        shearSlider.addEventListener('input', (e) => {
            shearValue.textContent = `${e.target.value}%`;
        });
        
        // Submit button
        const submitButton = document.getElementById('submitLabel');
        submitButton.addEventListener('click', () => this.submitLabel());
        
        // Reset button
        const resetButton = document.getElementById('resetButton');
        if (resetButton) {
            resetButton.addEventListener('click', () => this.resetInterface());
        }
        
        // Refresh buttons
        const refreshMetrics = document.getElementById('refreshMetrics');
        if (refreshMetrics) {
            refreshMetrics.addEventListener('click', () => this.loadMetrics());
        }
        
        const refreshHistory = document.getElementById('refreshHistory');
        if (refreshHistory) {
            refreshHistory.addEventListener('click', () => this.loadHistory());
        }
        
        // Export button
        const exportHistory = document.getElementById('exportHistory');
        if (exportHistory) {
            exportHistory.addEventListener('click', () => this.exportHistory());
        }
        
        // Live microscope event listeners
        this.initializeLiveMicroscopeListeners();
    }
    
    initializeLiveMicroscopeListeners() {
        // Camera control buttons
        const startCamera = document.getElementById('startCamera');
        const stopCamera = document.getElementById('stopCamera');
        const captureFrame = document.getElementById('captureFrame');
        const predictLive = document.getElementById('predictLive');
        
        if (startCamera) {
            startCamera.addEventListener('click', () => this.startCamera());
        }
        
        if (stopCamera) {
            stopCamera.addEventListener('click', () => this.stopCamera());
        }
        
        if (captureFrame) {
            captureFrame.addEventListener('click', () => this.captureFrame());
        }
        
        if (predictLive) {
            predictLive.addEventListener('click', () => this.predictLiveFrame());
        }
        
        // Detect cameras button
        const detectCameras = document.getElementById('detectCameras');
        if (detectCameras) {
            detectCameras.addEventListener('click', () => this.detectAvailableCameras());
        }
        
        // Auto prediction toggle
        const autoPrediction = document.getElementById('autoPrediction');
        if (autoPrediction) {
            autoPrediction.addEventListener('change', (e) => {
                this.toggleAutoPrediction(e.target.checked);
            });
        }
        
        // Prediction interval slider
        const predictionInterval = document.getElementById('predictionInterval');
        const intervalValue = document.getElementById('intervalValue');
        if (predictionInterval && intervalValue) {
            predictionInterval.addEventListener('input', (e) => {
                this.predictionIntervalSeconds = parseInt(e.target.value);
                intervalValue.textContent = `${e.target.value}s`;
                
                // Restart auto prediction if active
                if (this.autoPredictionEnabled) {
                    this.stopAutoPrediction();
                    this.startAutoPrediction();
                }
            });
        }
    }
    
    // Live Microscope Methods
    
    initializeLiveMicroscope() {
        // Initialize video feed element
        const videoFeed = document.getElementById('videoFeed');
        if (videoFeed) {
            videoFeed.onerror = () => {
                console.error('Video feed error');
                this.updateCameraStatus(false);
            };
        }
    }
    
    async checkCameraStatus() {
        try {
            const response = await fetch('/camera/status');
            if (response.ok) {
                const status = await response.json();
                this.updateCameraStatus(status.camera_active);
            }
        } catch (error) {
            console.error('Error checking camera status:', error);
            this.updateCameraStatus(false);
        }
    }
    
    updateCameraStatus(isActive) {
        this.cameraActive = isActive;
        
        const cameraStatus = document.getElementById('cameraStatus');
        const videoFeed = document.getElementById('videoFeed');
        const videoOverlay = document.getElementById('videoOverlay');
        const videoPlaceholder = document.getElementById('videoPlaceholder');
        const startCamera = document.getElementById('startCamera');
        const stopCamera = document.getElementById('stopCamera');
        const captureFrame = document.getElementById('captureFrame');
        const predictLive = document.getElementById('predictLive');
        
        if (isActive) {
            // Camera is active
            cameraStatus.className = 'camera-status camera-active';
            cameraStatus.innerHTML = '<i class="fas fa-video me-2"></i>Camera Active';
            
            // Show video feed
            if (videoFeed) {
                videoFeed.src = '/video_feed?' + new Date().getTime(); // Add timestamp to prevent caching
                videoFeed.style.display = 'block';
            }
            if (videoOverlay) videoOverlay.style.display = 'block';
            if (videoPlaceholder) videoPlaceholder.style.display = 'none';
            
            // Update button visibility
            if (startCamera) startCamera.style.display = 'none';
            if (stopCamera) stopCamera.style.display = 'inline-block';
            if (captureFrame) captureFrame.style.display = 'inline-block';
            if (predictLive) predictLive.style.display = 'inline-block';
        } else {
            // Camera is inactive
            cameraStatus.className = 'camera-status camera-inactive';
            cameraStatus.innerHTML = '<i class="fas fa-video-slash me-2"></i>Camera Inactive';
            
            // Hide video feed
            if (videoFeed) {
                videoFeed.style.display = 'none';
                videoFeed.src = '';
            }
            if (videoOverlay) videoOverlay.style.display = 'none';
            if (videoPlaceholder) videoPlaceholder.style.display = 'block';
            
            // Update button visibility
            if (startCamera) startCamera.style.display = 'inline-block';
            if (stopCamera) stopCamera.style.display = 'none';
            if (captureFrame) captureFrame.style.display = 'none';
            if (predictLive) predictLive.style.display = 'none';
            
            // Stop auto prediction if active
            this.stopAutoPrediction();
        }
    }
    
    async detectAvailableCameras() {
        try {
            this.showLoading('Detecting available cameras...');
            
            const response = await fetch('/camera/detect');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                const cameras = result.available_cameras;
                
                if (cameras.length === 0) {
                    alert('No cameras detected on the system.\n\nPlease check:\n• Camera is connected\n• Camera drivers are installed\n• Camera is not being used by another application');
                } else {
                    // Update camera dropdown with detected cameras
                    const cameraSelect = document.getElementById('cameraId');
                    cameraSelect.innerHTML = '';
                    
                    cameras.forEach(cam => {
                        const option = document.createElement('option');
                        option.value = cam.camera_id;
                        option.textContent = `Camera ${cam.camera_id} (${cam.width}x${cam.height})`;
                        cameraSelect.appendChild(option);
                    });
                    
                    // Show success message
                    const cameraList = cameras.map(cam => 
                        `Camera ${cam.camera_id}: ${cam.width}x${cam.height} @ ${cam.fps.toFixed(1)}fps`
                    ).join('\n');
                    
                    alert(`Found ${cameras.length} working camera(s):\n\n${cameraList}\n\nCamera dropdown has been updated.`);
                }
            } else {
                throw new Error(result.message || 'Failed to detect cameras');
            }
            
        } catch (error) {
            console.error('Error detecting cameras:', error);
            alert(`Error detecting cameras: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async startCamera() {
        try {
            const cameraId = document.getElementById('cameraId').value;
            this.showLoading('Starting camera...');
            
            const response = await fetch(`/camera/start?camera_id=${parseInt(cameraId)}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (response.ok && result.status === 'success') {
                this.updateCameraStatus(true);
                // Small delay to ensure camera is ready
                setTimeout(() => {
                    this.checkCameraStatus();
                }, 1000);
                
                // Show success message
                alert(`Camera ${cameraId} started successfully!`);
            } else {
                // Show detailed error message from server
                const errorMsg = result.detail || result.message || 'Failed to start camera';
                throw new Error(errorMsg);
            }
            
        } catch (error) {
            console.error('Error starting camera:', error);
            
            // Show user-friendly error message with suggestions
            let userMessage = `Failed to start camera: ${error.message}\n\n`;
            userMessage += 'Troubleshooting tips:\n';
            userMessage += '• Try clicking "Detect Available Cameras" first\n';
            userMessage += '• Make sure no other applications are using the camera\n';
            userMessage += '• Try different camera IDs (0, 1, 2, etc.)\n';
            userMessage += '• Check camera connections and drivers\n';
            userMessage += '• Restart the browser if needed';
            
            alert(userMessage);
            this.updateCameraStatus(false);
        } finally {
            this.hideLoading();
        }
    }
    
    async stopCamera() {
        try {
            this.showLoading('Stopping camera...');
            
            const response = await fetch('/camera/stop', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.updateCameraStatus(false);
            } else {
                throw new Error(result.message || 'Failed to stop camera');
            }
            
        } catch (error) {
            console.error('Error stopping camera:', error);
            alert(`Error stopping camera: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    async captureFrame() {
        try {
            this.showLoading('Capturing frame...');
            
            const response = await fetch('/camera/capture', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // Create a temporary image element to display the captured frame
                const modal = document.createElement('div');
                modal.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.8);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 10000;
                `;
                
                modal.innerHTML = `
                    <div style="background: white; padding: 20px; border-radius: 10px; max-width: 90%; max-height: 90%; overflow: auto;">
                        <h5>Captured Frame</h5>
                        <img src="${result.image_data}" style="max-width: 100%; height: auto; border-radius: 5px;">
                        <div style="text-align: center; margin-top: 15px;">
                            <button class="btn btn-primary me-2" onclick="this.parentElement.parentElement.parentElement.remove()">Close</button>
                            <button class="btn btn-success" onclick="mlmcscInterface.useFrameForPrediction('${result.image_data}'); this.parentElement.parentElement.parentElement.remove();">Use for Prediction</button>
                        </div>
                        <small class="text-muted d-block mt-2">Captured: ${new Date(result.timestamp).toLocaleString()}</small>
                    </div>
                `;
                
                document.body.appendChild(modal);
                
                // Close modal when clicking outside
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        modal.remove();
                    }
                });
                
            } else {
                throw new Error(result.message || 'Failed to capture frame');
            }
            
        } catch (error) {
            console.error('Error capturing frame:', error);
            alert(`Error capturing frame: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    async useFrameForPrediction(imageData) {
        // Use captured frame for prediction in the main interface
        this.currentImageData = imageData;
        
        try {
            this.showLoading('Analyzing captured frame...');
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_data: imageData,
                    image_format: 'jpg'
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const prediction = await response.json();
            this.displayPrediction(prediction);
            
        } catch (error) {
            console.error('Error analyzing captured frame:', error);
            alert(`Error analyzing frame: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    async predictLiveFrame() {
        try {
            this.showLoading('Running live prediction...');
            
            const response = await fetch('/camera/predict_live', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.displayLivePrediction(result.prediction);
            } else {
                throw new Error(result.message || 'Failed to run live prediction');
            }
            
        } catch (error) {
            console.error('Error in live prediction:', error);
            alert(`Error in live prediction: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    displayLivePrediction(prediction) {
        const livePredictionValue = document.getElementById('livePredictionValue');
        const livePredictionConfidence = document.getElementById('livePredictionConfidence');
        const liveSpecimenId = document.getElementById('liveSpecimenId');
        const livePredictionTime = document.getElementById('livePredictionTime');
        
        if (livePredictionValue) {
            livePredictionValue.textContent = `${prediction.shear_percentage.toFixed(1)}%`;
        }
        
        if (livePredictionConfidence) {
            const confidencePercent = (prediction.confidence * 100).toFixed(1);
            livePredictionConfidence.textContent = `${confidencePercent}%`;
            
            // Update confidence color
            if (prediction.confidence >= 0.8) {
                livePredictionConfidence.className = 'fs-6 text-success';
            } else if (prediction.confidence >= 0.6) {
                livePredictionConfidence.className = 'fs-6 text-warning';
            } else {
                livePredictionConfidence.className = 'fs-6 text-danger';
            }
        }
        
        if (liveSpecimenId) {
            liveSpecimenId.textContent = prediction.specimen_id;
        }
        
        if (livePredictionTime) {
            livePredictionTime.textContent = `Last prediction: ${new Date().toLocaleTimeString()}`;
        }
    }
    
    toggleAutoPrediction(enabled) {
        this.autoPredictionEnabled = enabled;
        
        if (enabled && this.cameraActive) {
            this.startAutoPrediction();
        } else {
            this.stopAutoPrediction();
        }
    }
    
    startAutoPrediction() {
        if (this.autoPredictionInterval) {
            clearInterval(this.autoPredictionInterval);
        }
        
        this.autoPredictionInterval = setInterval(() => {
            if (this.cameraActive && this.autoPredictionEnabled) {
                this.predictLiveFrame();
            }
        }, this.predictionIntervalSeconds * 1000);
    }
    
    stopAutoPrediction() {
        if (this.autoPredictionInterval) {
            clearInterval(this.autoPredictionInterval);
            this.autoPredictionInterval = null;
        }
        
        // Update checkbox state
        const autoPrediction = document.getElementById('autoPrediction');
        if (autoPrediction) {
            autoPrediction.checked = false;
        }
        this.autoPredictionEnabled = false;
    }

    async handleImageUpload(file) {
        try {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                alert('Please select a valid image file.');
                return;
            }
            
            // Show loading state
            this.showLoading('Analyzing image...');
            
            // Convert file to base64
            const imageData = await this.fileToBase64(file);
            this.currentImageData = imageData;
            
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_data: imageData,
                    image_format: file.type.split('/')[1]
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const prediction = await response.json();
            this.displayPrediction(prediction);
            
        } catch (error) {
            console.error('Error processing image:', error);
            alert(`Error processing image: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }
    
    displayPrediction(prediction) {
        this.currentPrediction = prediction;
        
        // Show image with overlay
        const imageDisplay = document.getElementById('imageDisplay');
        const analysisImage = document.getElementById('analysisImage');
        
        analysisImage.src = prediction.overlay_image;
        imageDisplay.style.display = 'block';
        
        // Update prediction display
        const modelPrediction = document.getElementById('modelPrediction');
        const modelConfidence = document.getElementById('modelConfidence');
        const specimenId = document.getElementById('specimenId');
        const confidenceIndicator = document.getElementById('confidenceIndicator');
        
        modelPrediction.textContent = prediction.shear_percentage.toFixed(1);
        modelConfidence.textContent = (prediction.confidence * 100).toFixed(1);
        specimenId.textContent = prediction.specimen_id;
        
        // Set confidence indicator color
        const confidencePercent = prediction.confidence * 100;
        confidenceIndicator.className = 'confidence-indicator';
        if (confidencePercent >= 80) {
            confidenceIndicator.classList.add('confidence-high');
        } else if (confidencePercent >= 60) {
            confidenceIndicator.classList.add('confidence-medium');
        } else {
            confidenceIndicator.classList.add('confidence-low');
        }
        
        // Show controls
        const controls = document.getElementById('controls');
        controls.style.display = 'block';
        
        // Set slider to model prediction as starting point
        const shearSlider = document.getElementById('shearSlider');
        const shearValue = document.getElementById('shearValue');
        shearSlider.value = Math.round(prediction.shear_percentage);
        shearValue.textContent = `${Math.round(prediction.shear_percentage)}%`;
    }
    
    async submitLabel() {
        if (!this.currentPrediction || !this.currentImageData) {
            alert('No prediction available to label.');
            return;
        }
        
        try {
            const shearSlider = document.getElementById('shearSlider');
            const technicianLabel = parseFloat(shearSlider.value);
            
            const submission = {
                specimen_id: this.currentPrediction.specimen_id,
                image_data: this.currentImageData,
                technician_label: technicianLabel,
                model_prediction: this.currentPrediction.shear_percentage,
                model_confidence: this.currentPrediction.confidence,
                technician_id: this.technicianId,
                notes: null
            };
            
            this.showLoading('Submitting label...');
            
            const response = await fetch('/submit_label', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(submission)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Show success message
            alert('Label submitted successfully!');
            
            // Refresh metrics and history
            this.loadMetrics();
            this.loadHistory();
            
            // Reset interface
            this.resetInterface();
            
        } catch (error) {
            console.error('Error submitting label:', error);
            alert(`Error submitting label: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    async loadMetrics() {
        try {
            const response = await fetch('/get_metrics');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const metrics = await response.json();
            this.displayMetrics(metrics);
            
        } catch (error) {
            console.error('Error loading metrics:', error);
        }
    }
    
    displayMetrics(metrics) {
        const metricsGrid = document.getElementById('metricsGrid');
        
        const metricsHtml = `
            <div class="col-md-4 col-sm-6 mb-3">
                <div class="metric-card">
                    <div class="metric-label">Total Predictions</div>
                    <div class="metric-value" style="color: #3498db;">
                        ${metrics.total_predictions}
                    </div>
                    <i class="fas fa-chart-bar fa-2x text-muted"></i>
                </div>
            </div>
            
            <div class="col-md-4 col-sm-6 mb-3">
                <div class="metric-card">
                    <div class="metric-label">Total Labels</div>
                    <div class="metric-value" style="color: #27ae60;">
                        ${metrics.total_labels}
                    </div>
                    <i class="fas fa-tags fa-2x text-muted"></i>
                </div>
            </div>
            
            <div class="col-md-4 col-sm-6 mb-3">
                <div class="metric-card">
                    <div class="metric-label">Mean Absolute Error</div>
                    <div class="metric-value" style="color: ${this.getErrorColor(metrics.accuracy_metrics.mae)};">
                        ${metrics.accuracy_metrics.mae ? metrics.accuracy_metrics.mae.toFixed(2) + '%' : 'N/A'}
                    </div>
                    <i class="fas fa-bullseye fa-2x text-muted"></i>
                </div>
            </div>
            
            <div class="col-md-4 col-sm-6 mb-3">
                <div class="metric-card">
                    <div class="metric-label">Root Mean Square Error</div>
                    <div class="metric-value" style="color: ${this.getErrorColor(metrics.accuracy_metrics.rmse)};">
                        ${metrics.accuracy_metrics.rmse ? metrics.accuracy_metrics.rmse.toFixed(2) + '%' : 'N/A'}
                    </div>
                    <i class="fas fa-square-root-alt fa-2x text-muted"></i>
                </div>
            </div>
            
            <div class="col-md-4 col-sm-6 mb-3">
                <div class="metric-card">
                    <div class="metric-label">Model Version</div>
                    <div class="metric-value" style="color: #6c757d; font-size: 1.8rem;">
                        ${metrics.model_version}
                    </div>
                    <i class="fas fa-code-branch fa-2x text-muted"></i>
                </div>
            </div>
            
            <div class="col-md-4 col-sm-6 mb-3">
                <div class="metric-card">
                    <div class="metric-label">Last Updated</div>
                    <div style="font-size: 1rem; color: #6c757d; margin-top: 10px;">
                        ${new Date(metrics.last_updated).toLocaleString()}
                    </div>
                    <i class="fas fa-clock fa-2x text-muted"></i>
                </div>
            </div>
        `;
        
        metricsGrid.innerHTML = metricsHtml;
    }
    
    getErrorColor(error) {
        if (!error) return '#6c757d';
        if (error < 5) return '#28a745';  // Green for low error
        if (error < 10) return '#ffc107'; // Yellow for medium error
        return '#dc3545'; // Red for high error
    }
    
    async loadHistory(page = 1) {
        try {
            const response = await fetch(`/get_history?page=${page}&page_size=10`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const history = await response.json();
            this.displayHistory(history);
            
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }
    
    displayHistory(history) {
        const historyBody = document.getElementById('historyBody');
        
        if (history.items.length === 0) {
            historyBody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No labeling history available</td></tr>';
            return;
        }
        
        const historyHtml = history.items.map(item => `
            <tr>
                <td>${new Date(item.timestamp).toLocaleString()}</td>
                <td><span class="badge bg-secondary">${item.specimen_id}</span></td>
                <td><span class="badge bg-primary">${item.technician_label.toFixed(1)}%</span></td>
                <td><span class="badge bg-info">${item.model_prediction.toFixed(1)}%</span></td>
                <td>
                    <span class="difference-badge ${this.getDifferenceBadgeClass(item.difference)}">
                        ${item.difference.toFixed(1)}%
                    </span>
                </td>
                <td><span class="badge bg-dark">${item.technician_id}</span></td>
            </tr>
        `).join('');
        
        historyBody.innerHTML = historyHtml;
        
        // Update pagination
        this.updatePagination(history);
    }
    
    getDifferenceBadgeClass(difference) {
        if (difference < 5) return 'bg-success';
        if (difference < 10) return 'bg-warning';
        return 'bg-danger';
    }
    
    updatePagination(history) {
        const pagination = document.getElementById('historyPagination');
        if (!pagination) return;
        
        const totalPages = Math.ceil(history.total_count / history.page_size);
        const currentPage = history.page;
        
        if (totalPages <= 1) {
            pagination.innerHTML = '';
            return;
        }
        
        let paginationHtml = '';
        
        // Previous button
        if (currentPage > 1) {
            paginationHtml += `
                <li class="page-item">
                    <a class="page-link" href="#" onclick="mlmcscInterface.loadHistory(${currentPage - 1})">Previous</a>
                </li>
            `;
        }
        
        // Page numbers
        const startPage = Math.max(1, currentPage - 2);
        const endPage = Math.min(totalPages, currentPage + 2);
        
        for (let i = startPage; i <= endPage; i++) {
            paginationHtml += `
                <li class="page-item ${i === currentPage ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="mlmcscInterface.loadHistory(${i})">${i}</a>
                </li>
            `;
        }
        
        // Next button
        if (currentPage < totalPages) {
            paginationHtml += `
                <li class="page-item">
                    <a class="page-link" href="#" onclick="mlmcscInterface.loadHistory(${currentPage + 1})">Next</a>
                </li>
            `;
        }
        
        pagination.innerHTML = paginationHtml;
    }
    
    async exportHistory() {
        try {
            this.showLoading('Exporting history...');
            
            const response = await fetch('/export_history');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `labeling_history_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
        } catch (error) {
            console.error('Error exporting history:', error);
            alert(`Error exporting history: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    getDifferenceColor(difference) {
        if (difference < 5) return '#28a745';  // Green for small difference
        if (difference < 10) return '#ffc107'; // Yellow for medium difference
        return '#dc3545'; // Red for large difference
    }
    
    resetInterface() {
        // Hide image and controls
        document.getElementById('imageDisplay').style.display = 'none';
        document.getElementById('controls').style.display = 'none';
        
        // Reset file input
        document.getElementById('imageInput').value = '';
        
        // Clear current data
        this.currentPrediction = null;
        this.currentImageData = null;
    }
    
    showLoading(message) {
        // Create or update loading overlay
        let overlay = document.getElementById('loadingOverlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loadingOverlay';
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                color: white;
                font-size: 18px;
            `;
            document.body.appendChild(overlay);
        }
        
        overlay.innerHTML = `
            <div style="text-align: center;">
                <div style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 2s linear infinite; margin: 0 auto 20px;"></div>
                <div>${message}</div>
            </div>
        `;
        
        // Add CSS animation if not already added
        if (!document.getElementById('spinAnimation')) {
            const style = document.createElement('style');
            style.id = 'spinAnimation';
            style.textContent = `
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
        }
        
        overlay.style.display = 'flex';
    }
    
    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
}

// Global reference for pagination
let mlmcscInterface;

// Initialize the interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    mlmcscInterface = new MLMCSCInterface();
});
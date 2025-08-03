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
        
        this.initializeEventListeners();
        this.loadMetrics();
        this.loadHistory();
        
        // Auto-refresh metrics and history every 30 seconds
        setInterval(() => {
            this.loadMetrics();
            this.loadHistory();
        }, 30000);
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
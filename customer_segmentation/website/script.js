const API_BASE_URL = 'http://localhost:5000';

// Configure axios defaults
axios.defaults.baseURL = API_BASE_URL;

function showHome() {
    const mainContent = document.getElementById("main-content");
    mainContent.innerHTML = `
        <section class="home">
            <h2>Welcome to the Customer Insights and Sales Prediction Dashboard</h2>
            <p>Navigate through the features using the menu above to explore insights, predictions, and analysis tools.</p>
            <div class="feature-overview">
                <h3>Available Features:</h3>
                <ul>
                    <li>Data Upload - Upload your customer and sales data</li>
                    <li>Customer Segmentation - Analyze customer groups</li>
                    <li>Sales Forecasting - Predict future sales trends</li>
                    <li>Sentiment Analysis - Analyze customer feedback</li>
                    <li>Regression Analysis - Compare different prediction models</li>
                </ul>
            </div>
        </section>
    `;
}

function showUploadData() {
    const mainContent = document.getElementById("main-content");
    mainContent.innerHTML = `
        <section class="upload">
            <h2>Upload Customer or Sales Data</h2>
            <div class="upload-container">
                <input type="file" id="fileInput" accept=".csv,.xlsx" />
                <button onclick="uploadFile()">Upload</button>
            </div>
            <div id="uploadStatus"></div>
            <div id="datasetPreview" class="result"></div>
        </section>
    `;
}

async function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const uploadStatus = document.getElementById("uploadStatus");
    const datasetPreview = document.getElementById("datasetPreview");
    const file = fileInput.files[0];

    if (!file) {
        uploadStatus.innerHTML = '<p class="error">Please select a file to upload.</p>';
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        uploadStatus.innerHTML = '<p>Uploading...</p>';
        const response = await axios.post("/upload", formData, {
            headers: { "Content-Type": "multipart/form-data" },
        });
        uploadStatus.innerHTML = `<p class="success">File uploaded successfully: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(2)} KB)</p>`;
        datasetPreview.innerHTML = `
            <h3>Uploaded Dataset Preview:</h3>
            <p><strong>Columns:</strong> ${response.data.columns.join(", ")}</p>
            <p><strong>Total Rows:</strong> ${response.data.rowCount}</p>
        `;
        sessionStorage.setItem("datasetUploaded", true);
    } catch (error) {
        uploadStatus.innerHTML = `<p class="error">Error uploading file: ${error.response?.data?.error || error.message}</p>`;
        console.error('Upload error:', error);
    }
}


function showSegmentation() {
    const mainContent = document.getElementById("main-content");
    mainContent.innerHTML = `
        <section class="segmentation">
            <h2>Customer Segmentation</h2>
            <div class="segmentation-options">
                <h3>Select Segmentation Approach:</h3>
                <select id="segmentationType" onchange="updateFeatureSelection()">
                    <option value="behavioral">Behavioral Segmentation</option>
                    <option value="engagement">Engagement Segmentation</option>
                    <option value="custom">Custom Selection</option>
                </select>
            </div>
            <div id="featureSelection" style="margin: 20px 0;">
                <p>Loading available features...</p>
            </div>
            <button onclick="performSegmentation()" id="segmentButton" disabled>Perform Segmentation</button>
            <div id="segmentationResults"></div>
        </section>
    `;
    
    loadAvailableFeatures();
}

function updateFeatureSelection() {
    const segmentationType = document.getElementById('segmentationType').value;
    const checkboxes = document.querySelectorAll('input[name="features"]');
    
    checkboxes.forEach(checkbox => checkbox.checked = false);
    
    switch(segmentationType) {
        case 'behavioral':
            selectFeatures(['Order_Frequency', 'Abandoned_Carts']);
            break;
        case 'engagement':
            selectFeatures(['Email_Interactions', 'Social_Media_Activity']);
            break;
    }
    
    updateSegmentButton();
}

function selectFeatures(featureNames) {
    featureNames.forEach(name => {
        const checkbox = document.querySelector(`input[value="${name}"]`);
        if (checkbox) checkbox.checked = true;
    });
}

async function loadAvailableFeatures() {
    const featureSelection = document.getElementById("featureSelection");
    
    const numericalFeatures = [
        'Age',
        'Product_Views',
        'Abandoned_Carts',
        'Order_Frequency',
        'Email_Interactions',
        'Ad_Engagement',
        'Social_Media_Activity',
        'Loyalty_Program_Participation',
        'sales'
    ];
    
    try {
        const response = await axios.get("/upload");
        const columns = response.data.columns;
        const availableFeatures = columns.filter(col => numericalFeatures.includes(col));
        
        featureSelection.innerHTML = `
            <div class="features-container">
                <p>Select two features for segmentation:</p>
                <div class="features-list">
                    ${availableFeatures.map(column => `
                        <div class="feature-item">
                            <input type="checkbox" id="${column}" name="features" value="${column}">
                            <label for="${column}">${column}</label>
                        </div>
                    `).join('')}
                </div>
                <p id="featureWarning" class="warning"></p>
            </div>
        `;
        
        document.querySelectorAll('input[name="features"]').forEach(checkbox => {
            checkbox.addEventListener('change', updateSegmentButton);
        });
        
    } catch (error) {
        featureSelection.innerHTML = `
            <p class="error">Error: ${error.response?.data?.error || 'No data available. Please upload data first.'}</p>
        `;
    }
}

function updateSegmentButton() {
    const checked = document.querySelectorAll('input[name="features"]:checked');
    const warning = document.getElementById('featureWarning');
    const segmentButton = document.getElementById('segmentButton');
    
    if (checked.length > 2) {
        warning.textContent = 'Please select only two features.';
        segmentButton.disabled = true;
    } else {
        warning.textContent = checked.length < 2 ? 'Please select two features.' : '';
        segmentButton.disabled = checked.length !== 2;
    }
}

async function performSegmentation() {
    const segmentationResults = document.getElementById("segmentationResults");
    const selectedFeatures = Array.from(document.querySelectorAll('input[name="features"]:checked'))
        .map(checkbox => checkbox.value);
    
    if (selectedFeatures.length !== 2) {
        segmentationResults.innerHTML = '<p class="error">Please select exactly two features.</p>';
        return;
    }
    
    segmentationResults.innerHTML = '<p>Running segmentation analysis...</p>';
    
    try {
        const response = await axios.post("/segmentation", { features: selectedFeatures });
        if (response.data.plot) {
            segmentationResults.innerHTML = `
                <h3>Segmentation Results</h3>
                <img src="data:image/png;base64,${response.data.plot}" alt="Segmentation Plot" class="result-plot">
                <div class="segmentation-explanation">
                    <h4>Analysis of ${selectedFeatures.join(' vs ')} Segmentation:</h4>
                    <p>The plot shows customer segments based on ${selectedFeatures.join(' and ')}.</p>
                    <ul>
                        <li>Cluster 0: Low ${selectedFeatures[0]} and Low ${selectedFeatures[1]}</li>
                        <li>Cluster 1: High ${selectedFeatures[0]} and High ${selectedFeatures[1]}</li>
                        <li>Cluster 2: Mixed or Moderate values</li>
                    </ul>
                </div>
            `;
        } else {
            segmentationResults.innerHTML = `<pre>${JSON.stringify(response.data, null, 2)}</pre>`;
        }
    } catch (error) {
        segmentationResults.innerHTML = `<p class="error">Error: ${error.response?.data?.error || error.message}</p>`;
        console.error('Segmentation error:', error);
    }
}

function showSalesPrediction() {
    const mainContent = document.getElementById("main-content");
    mainContent.innerHTML = `
        <section class="sales-prediction">
            <h2>Sales and Order Frequency Analysis</h2>
            <div class="forecast-options">
                <div class="analysis-info">
                    <h3>Analysis Components:</h3>
                    <ul>
                        <li>Sales data trends</li>
                        <li>Order frequency patterns</li>
                        <li>Combined performance metrics</li>
                        <li>30-day future forecast</li>
                    </ul>
                </div>
                <button onclick="predictSales()" class="forecast-button">Generate Forecast Analysis</button>
            </div>
            <div id="salesResults"></div>
        </section>
    `;
}

async function predictSales() {
    const salesResults = document.getElementById("salesResults");
    salesResults.innerHTML = `
        <div class="loading-message">
            <p>Analyzing sales and order patterns...</p>
            <p>Generating comprehensive forecast...</p>
        </div>
    `;

    try {
        const response = await axios.post("/sales-prediction");
        if (response.data.plot) {
            salesResults.innerHTML = `
                <div class="forecast-results">
                    <h3>Forecast Analysis Results</h3>
                    
                    <div class="forecast-plot">
                        <img src="data:image/png;base64,${response.data.plot}" alt="Sales Forecast Plot" class="result-plot">
                    </div>
                    
                    
                    <div class="forecast-interpretation">
                        <h4>Key Insights</h4>
                        <ul>
                            <li>The blue line shows historical performance</li>
                            <li>The red dashed line shows the model's predictions</li>
                            <li>The green dashed line shows the 30-day forecast</li>
                        </ul>
                    </div>
                </div>
            `;
        } else {
            salesResults.innerHTML = `
                <div class="error-message">
                    <p>Unable to generate forecast visualization.</p>
                    <pre>${JSON.stringify(response.data, null, 2)}</pre>
                </div>
            `;
        }
    } catch (error) {
        salesResults.innerHTML = `
            <div class="error-message">
                <h4>Error in Forecast Generation</h4>
                <p>${error.response?.data?.error || error.message}</p>
                <div class="troubleshooting-tips">
                    <h4>Troubleshooting Tips:</h4>
                    <ul>
                        <li>Ensure your data includes both 'sales' and 'Order_Frequency' columns</li>
                        <li>Check that the data contains sufficient historical information</li>
                        <li>Verify that the values are numerical</li>
                    </ul>
                </div>
            </div>
        `;
        console.error('Prediction error:', error);
    }
}

function showSentimentAnalysis() {
    const mainContent = document.getElementById("main-content");
    mainContent.innerHTML = `
        <section class="sentiment-analysis">
            <h2>Customer Sentiment Analysis</h2>
            <div class="sentiment-input">
                <input type="number" id="customerIdInput" placeholder="Enter Customer ID" min="1" step="1" required />
                <button onclick="analyzeSentiment()" class="analyze-button">Analyze Sentiment</button>
            </div>
            <div id="sentimentResults"></div>
        </section>
    `;
}

async function analyzeSentiment() {
    const customerIdInput = document.getElementById("customerIdInput");
    const sentimentResults = document.getElementById("sentimentResults");
    
    // Clear previous results
    sentimentResults.innerHTML = '';
    
    // Validate input
    if (!customerIdInput.value || customerIdInput.value.trim() === '') {
        sentimentResults.innerHTML = '<p class="error">Please enter a Customer ID.</p>';
        return;
    }

    // Convert to integer and validate
    const customerId = parseInt(customerIdInput.value);
    if (isNaN(customerId) || customerId <= 0) {
        sentimentResults.innerHTML = '<p class="error">Please enter a valid positive Customer ID.</p>';
        return;
    }

    // Show loading state
    sentimentResults.innerHTML = '<p class="loading">Analyzing customer sentiment...</p>';

    try {
        const response = await axios.post("/sentiment-analysis", {
            Customer_ID: customerId
        });

        if (response.data && response.data.sentiment_score) {
            const { Customer_ID, Order_Frequency, sentiment_label } = response.data.sentiment_score;
            
            // Define sentiment colors
            const sentimentColors = {
                'Positive': '#28a745',
                'Neutral': '#ffc107',
                'Negative': '#dc3545'
            };

            sentimentResults.innerHTML = `
                <div class="sentiment-results-card">
                    <h3>Sentiment Analysis Results</h3>
                    <div class="sentiment-details">
                        <p><strong>Customer ID:</strong> ${Customer_ID}</p>
                        <p><strong>Order Frequency:</strong> ${Order_Frequency}</p>
                        <p><strong>Sentiment:</strong> 
                            <span style="color: ${sentimentColors[sentiment_label]}; font-weight: bold;">
                                ${sentiment_label}
                            </span>
                        </p>
                    </div>
                    <div class="sentiment-explanation">
                        <p><em>Based on order frequency:</em></p>
                        <ul>
                            <li>Less than 4 orders: Negative sentiment</li>
                            <li>4-7 orders: Neutral sentiment</li>
                            <li>More than 7 orders: Positive sentiment</li>
                        </ul>
                    </div>
                </div>
            `;
        } else {
            throw new Error('Invalid response format');
        }
    } catch (error) {
        const errorMessage = error.response?.data?.error || error.message || 'An unknown error occurred';
        sentimentResults.innerHTML = `
            <div class="error-card">
                <p class="error">Error: ${errorMessage}</p>
                <p class="error-help">Please check the Customer ID and try again.</p>
            </div>
        `;
        console.error("Sentiment analysis error:", error);
    }
}

function showRegressionComparison() {
    const mainContent = document.getElementById("main-content");
    mainContent.innerHTML = `
        <section class="regression-comparison">
            <h2>Regression Models Comparison</h2>
            <button onclick="compareModels()">Compare Models</button>
            <div id="comparisonResults"></div>
        </section>
    `;
}

async function compareModels() {
    const comparisonResults = document.getElementById("comparisonResults");
    comparisonResults.innerHTML = '<p>Comparing regression models...</p>';

    try {
        const response = await axios.post("/regression-comparison", { 
            features: ["Age", "Order_Frequency", "Product_Views"], 
            target: "Purchase_History" 
        });
        
        const results = response.data.results;
        comparisonResults.innerHTML = `
            <h3>Model Comparison Results</h3>
            <div class="model-results">
                ${Object.entries(results).map(([model, metrics]) => `
                    <div class="model-card">
                        <h4>${model}</h4>
                        <p>MSE: ${metrics.MSE.toFixed(4)}</p>
                        <p>R² Score: ${metrics.R2.toFixed(4)}</p>
                    </div>
                `).join('')}
            </div>
        `;
    } catch (error) {
        const errorMessage = error.response 
            ? `Error: ${error.response.data?.error} (Status: ${error.response.status})`
            : `Error: ${error.message}`;
        
        comparisonResults.innerHTML = `<p class="error">${errorMessage}</p>`;
        console.error('Model comparison error:', error);
    }
}


// Initialize home page when document loads
document.addEventListener("DOMContentLoaded", () => {
    showHome();
});
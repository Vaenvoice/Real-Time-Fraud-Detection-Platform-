document.addEventListener('DOMContentLoaded', () => {
    const predictionForm = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultDisplay = document.getElementById('result-display');
    const alertFeed = document.getElementById('alert-feed');
    const serverStatus = document.getElementById('server-status');
    const loader = document.querySelector('.loader-spinner');
    const btnText = document.querySelector('.btn-text');

    // API Configuration
    const API_URL = '';

    // Initialize Chart
    const ctx = document.getElementById('riskChart').getContext('2d');
    const riskChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['12am', '4am', '8am', '12pm', '4pm', '8pm', '11pm'],
            datasets: [{
                label: 'Fraud Probability Score',
                data: [12, 19, 3, 5, 2, 3, 7],
                borderColor: '#ffffff',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#a0a0c0' } },
                x: { grid: { display: false }, ticks: { color: '#a0a0c0' } }
            }
        }
    });

    // Handle Form Submission
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const amount = document.getElementById('amount').value;
        const v1 = document.getElementById('v1').value || 0.0;
        const v2 = document.getElementById('v2').value || 0.0;

        // Construct full transaction object to match backend schema
        const transactionData = {
            Amount: parseFloat(amount),
            Time: Date.now() % 100000, // Mock time
            V1: parseFloat(v1),
            V2: parseFloat(v2)
        };
        
        // Fill V3-V28 with defaults
        for (let i = 3; i <= 28; i++) {
            transactionData[`V${i}`] = 0.0;
        }

        // UI Loading State
        setLoading(true);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(transactionData)
            });

            if (!response.ok) throw new Error('API Error');

            const result = await response.json();
            displayResult(result);
            addToFeed(amount, result);
        } catch (error) {
            console.error('Prediction failed:', error);
            // Fallback for demo if API is not running
            mockPrediction(amount);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            predictBtn.disabled = true;
            btnText.style.opacity = '0';
            loader.classList.remove('hidden');
        } else {
            predictBtn.disabled = false;
            btnText.style.opacity = '1';
            loader.classList.add('hidden');
        }
    }

    function displayResult(result) {
        resultDisplay.classList.remove('hidden');
        const badge = document.getElementById('result-badge');
        const prob = document.getElementById('result-probability');
        const expl = document.getElementById('result-explanation');

        const isFraud = result.fraud_label === 1;
        badge.textContent = isFraud ? 'FRAUD ' : 'SAFE';
        badge.className = `badge ${isFraud ? 'badge-danger' : 'badge-safe'}`;
        
        prob.textContent = `Probability: ${(result.fraud_probability * 100).toFixed(2)}%`;
        
        // Handle explanation if it's an object/dict
        if (typeof result.explanation === 'object') {
            const topFeature = Object.keys(result.explanation)[0] || "Unknown";
            expl.textContent = `Key risk factor detected in feature: ${topFeature}`;
        } else {
            expl.textContent = result.explanation;
        }

        if (isFraud) {
            resultDisplay.classList.add('danger-glow');
        } else {
            resultDisplay.classList.remove('danger-glow');
        }
    }

    function addToFeed(amount, result) {
        const item = document.createElement('div');
        const isFraud = result.fraud_label === 1;
        const id = Math.random().toString(36).substr(2, 6).toUpperCase();
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        item.className = `alert-item ${isFraud ? 'fraud-transaction danger-pulse' : 'safe-transaction'}`;
        item.innerHTML = `
            <div class="alert-info">
                <span class="transaction-id">ID: ${id}... <span class="time-stamp">${timeStr}</span></span>
                <span class="transaction-amount">$${parseFloat(amount).toFixed(2)}</span>
            </div>
            <div class="alert-status">${isFraud ? 'FRAUD' : 'SAFE'}</div>
        `;

        alertFeed.prepend(item);
        if (alertFeed.children.length > 8) {
            alertFeed.removeChild(alertFeed.lastChild);
        }
    }

    // Mock Prediction for Demo
    function mockPrediction(amount) {
        const prob = parseFloat(amount) > 1000 ? (0.7 + Math.random() * 0.3) : (Math.random() * 0.2);
        const result = {
            fraud_label: prob > 0.5 ? 1 : 0,
            fraud_probability: prob,
            explanation: prob > 0.5 ? "Suspicious amount detected by AI model." : "Transaction patterns within normal range."
        };
        displayResult(result);
        addToFeed(amount, result);
    }

    // Fetch Dashboard Stats
    async function updateDashboard() {
        try {
            const response = await fetch(`${API_URL}/stats`);
            if (response.ok) {
                const stats = await response.json();
                document.querySelector('#total-transactions .stat-value').textContent = stats.total_scanned.toLocaleString();
                document.querySelector('#fraud-detected .stat-value').textContent = stats.fraud_detected.toLocaleString();
                
                const avgRiskElem = document.querySelector('#avg-prob .stat-value');
                avgRiskElem.textContent = `${(stats.avg_risk_score * 100).toFixed(1)}%`;
                document.querySelector('.risk-bar').style.width = `${Math.min(stats.avg_risk_score * 100, 100)}%`;
            }
        } catch (e) {
            console.error("Failed to update dashboard:", e);
        }
    }

    // Fetch Recent Transactions for Feed & Chart
    async function updateFeedAndChart() {
        try {
            const response = await fetch(`${API_URL}/recent-transactions`);
            if (response.ok) {
                const transactions = await response.json();
                
                // Update Feed
                alertFeed.innerHTML = '';
                transactions.forEach(tx => {
                    const item = document.createElement('div');
                    const isFraud = tx.fraud_label === 1;
                    const localTime = new Date(tx.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

                    item.className = `alert-item ${isFraud ? 'fraud-transaction danger-pulse' : 'safe-transaction'}`;
                    item.innerHTML = `
                        <div class="alert-info">
                            <span class="transaction-id">ID: TX-${tx.id} <span class="time-stamp">${localTime}</span></span>
                            <span class="transaction-amount">$${tx.amount.toFixed(2)}</span>
                        </div>
                        <div class="alert-status">${isFraud ? 'FRAUD' : 'SAFE'}</div>
                    `;
                    alertFeed.appendChild(item);
                });

                // Update Chart
                if (transactions.length > 0) {
                    const chartData = transactions.slice(0, 10).reverse().map(tx => tx.fraud_probability * 100);
                    const chartLabels = transactions.slice(0, 10).reverse().map(tx => {
                        return new Date(tx.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    });
                    
                    riskChart.data.labels = chartLabels;
                    riskChart.data.datasets[0].data = chartData;
                    riskChart.update('none'); 
                }
            }
        } catch (e) {
            console.error("Failed to update feed:", e);
        }
    }

    // Check Server Health
    async function checkHealth() {
        try {
            const resp = await fetch(`${API_URL}/health`);
            if (resp.ok) {
                serverStatus.innerHTML = '<span class="status-dot pulse-green"></span> System Online';
            }
        } catch (e) {
            serverStatus.innerHTML = '<span class="status-dot" style="background:gray"></span> Demo Mode (Offline)';
        }
    }

    // Initial load
    checkHealth();
    updateDashboard();
    updateFeedAndChart();

    // Intervals
    setInterval(checkHealth, 30000);
    setInterval(updateDashboard, 5000);
    setInterval(updateFeedAndChart, 3000);
});

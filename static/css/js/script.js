// script.js

// Initialize statistics
let colorPredCount = 0;
let minePredCount = 0;
let colorHistory = [];
let mineHistory = [];

// Initialize minesweeper grid
function initMineGrid() {
    const grid = document.getElementById('mineGrid');
    grid.innerHTML = '';
    for (let i = 0; i < 25; i++) {
        const cell = document.createElement('div');
        cell.className = 'mine-cell';
        cell.dataset.index = i;
        grid.appendChild(cell);
    }
}

// Predict color game
async function predictColor() {
    const btn = document.getElementById('colorBtn');
    const loading = document.getElementById('colorLoading');
    const result = document.getElementById('colorResult');
    
    btn.disabled = true;
    loading.classList.add('show');
    result.classList.remove('show');

    // Clear previous predictions
    document.querySelectorAll('.color-box').forEach(box => {
        box.classList.remove('predicted');
    });

    try {
        const response = await fetch('/predict/color', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();
        
        setTimeout(() => {
            loading.classList.remove('show');
            result.classList.add('show');
            
            // Highlight predicted color
            const predictedBox = document.querySelector(`[data-color="${data.prediction}"]`);
            if (predictedBox) {
                predictedBox.classList.add('predicted');
            }

            // Update result
            document.getElementById('colorPrediction').innerHTML = `
                <strong>Predicted Color:</strong> ${data.prediction.toUpperCase()}<br>
                <strong>Confidence:</strong> ${data.confidence}%<br>
                <strong>Pattern:</strong> ${data.pattern}<br>
                <strong>Reasoning:</strong> ${data.reasoning}
            `;

            // Update stats
            colorPredCount++;
            document.getElementById('color-predictions').textContent = colorPredCount;
            document.getElementById('color-confidence').textContent = data.confidence + '%';
            document.getElementById('color-pattern').textContent = data.pattern;

            // Add to history
            addColorHistory(data);
            
            btn.disabled = false;
        }, 1500);

    } catch (error) {
        console.error('Error:', error);
        loading.classList.remove('show');
        alert('Error making prediction. Please check your API key and server connection.');
        btn.disabled = false;
    }
}

// Predict minesweeper
async function predictMines() {
    const btn = document.getElementById('mineBtn');
    const loading = document.getElementById('mineLoading');
    const result = document.getElementById('mineResult');
    
    btn.disabled = true;
    loading.classList.add('show');
    result.classList.remove('show');

    // Clear previous predictions
    document.querySelectorAll('.mine-cell').forEach(cell => {
        cell.className = 'mine-cell';
        cell.textContent = '';
    });

    try {
        const response = await fetch('/predict/minesweeper', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();
        
        setTimeout(() => {
            loading.classList.remove('show');
            result.classList.add('show');
            
            // Show safe and danger spots
            data.safe_spots.forEach(index => {
                const cell = document.querySelector(`[data-index="${index}"]`);
                if (cell) {
                    cell.classList.add('safe');
                    cell.textContent = '✓';
                }
            });

            data.danger_spots.forEach(index => {
                const cell = document.querySelector(`[data-index="${index}"]`);
                if (cell) {
                    cell.classList.add('danger');
                    cell.textContent = '💣';
                }
            });

            // Update result
            document.getElementById('minePrediction').innerHTML = `
                <strong>Safe Spots:</strong> ${data.safe_spots.length} locations<br>
                <strong>Danger Spots:</strong> ${data.danger_spots.length} locations<br>
                <strong>Confidence:</strong> ${data.confidence}%<br>
                <strong>Reasoning:</strong> ${data.reasoning}
            `;

            // Update stats
            minePredCount++;
            document.getElementById('mine-predictions').textContent = minePredCount;
            document.getElementById('mine-safe').textContent = data.safe_spots.length;
            document.getElementById('mine-confidence').textContent = data.confidence + '%';

            // Add to history
            addMineHistory(data);
            
            btn.disabled = false;
        }, 1500);

    } catch (error) {
        console.error('Error:', error);
        loading.classList.remove('show');
        alert('Error making prediction. Please check your API key and server connection.');
        btn.disabled = false;
    }
}

// Add to color history
function addColorHistory(data) {
    const historyList = document.getElementById('colorHistoryList');
    const time = new Date().toLocaleTimeString();
    const item = document.createElement('div');
    item.className = 'history-item';
    item.textContent = `${time} - ${data.prediction.toUpperCase()} (${data.confidence}%)`;
    historyList.insertBefore(item, historyList.firstChild);
    
    // Keep only last 5
    while (historyList.children.length > 5) {
        historyList.removeChild(historyList.lastChild);
    }
}

// Add to mine history
function addMineHistory(data) {
    const historyList = document.getElementById('mineHistoryList');
    const time = new Date().toLocaleTimeString();
    const item = document.createElement('div');
    item.className = 'history-item';
    item.textContent = `${time} - ${data.safe_spots.length} safe spots (${data.confidence}%)`;
    historyList.insertBefore(item, historyList.firstChild);
    
    // Keep only last 5
    while (historyList.children.length > 5) {
        historyList.removeChild(historyList.lastChild);
    }
}

// Initialize on load
window.addEventListener('DOMContentLoaded', function() {
    initMineGrid();
    console.log('Bingo Plus Game Predictor initialized');
    console.log('Remember: This is for educational purposes only!');
});

// Add keyboard shortcuts
document.addEventListener('keypress', function(e) {
    if (e.key === 'c' || e.key === 'C') {
        const colorBtn = document.getElementById('colorBtn');
        if (!colorBtn.disabled) {
            predictColor();
        }
    } else if (e.key === 'm' || e.key === 'M') {
        const mineBtn = document.getElementById('mineBtn');
        if (!mineBtn.disabled) {
            predictMines();
        }
    }
});
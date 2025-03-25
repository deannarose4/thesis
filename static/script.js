const firebaseConfig = {
	apiKey: "AIzaSyA7BY7CR36WZno91kn7gPCodBMGmorvDdQ",
	authDomain: "toggle-button-7c0d9.firebaseapp.com",
	databaseURL: "https://toggle-button-7c0d9-default-rtdb.asia-southeast1.firebasedatabase.app",
	projectId: "toggle-button-7c0d9",
	storageBucket: "toggle-button-7c0d9.firebasestorage.app",
	messagingSenderId: "443319352788",
	appId: "1:443319352788:web:6107fdb2a12a2af481ea41",
	measurementId: "G-CQXNPP44D7"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const database = firebase.database();

document.addEventListener('DOMContentLoaded', function() {
    const selectImageBtn = document.getElementById('selectImage');
    const fileUpload = document.getElementById('file-upload');
    const displayImage = document.getElementById('displayImage');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const detectedElement = document.getElementById('detected');
    const confidenceElement = document.getElementById('confidence');
    const resultIcon = document.getElementById('result-icon');
    
    if (selectImageBtn && fileUpload) {
        selectImageBtn.addEventListener('click', () => {
            fileUpload.click();
        });

        fileUpload.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                // Display the image
                const reader = new FileReader();
                reader.onload = (e) => {
                    displayImage.src = e.target.result;
                    displayImage.style.display = 'block';
                    uploadPlaceholder.style.display = 'none';
                };
                reader.readAsDataURL(file);

                // Reset results
                detectedElement.textContent = 'Analyzing...';
                confidenceElement.textContent = 'Please wait';
                resultIcon.innerHTML = '<i class="material-icons spinning">autorenew</i>';

                // Send to server for prediction
                const formData = new FormData();
                formData.append('file', file);

                try {
                    console.log('Sending image for prediction...');
                    const response = await fetch('https://upcyclerender.onrender.com/predict', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        if (response.status === 502) {
                            throw new Error('Server is temporarily unavailable. Please try again later.');
                        }
                        throw new Error(`Server error: ${response.status}`);
                    }

                    const data = await response.json();
                    console.log('Received prediction:', data);

                    // Update the UI with the prediction results
                    detectedElement.textContent = `Detected: ${data.prediction}`;
                    confidenceElement.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
                    
                    // Update result icon based on prediction
                    if (data.prediction === 'Biodegradable') {
                        resultIcon.innerHTML = '<i class="material-icons green-text">eco</i>';
                    } else {
                        resultIcon.innerHTML = '<i class="material-icons red-text">delete</i>';
                    }

                    // Save prediction to Firebase
                    const timestamp = new Date().toISOString();
                    database.ref('predictions').push({
                        prediction: data.prediction,
                        confidence: data.confidence,
                        timestamp: timestamp
                    });
                } catch (error) {
                    console.error('Error:', error);
                    detectedElement.textContent = 'Error: Failed to process image';
                    confidenceElement.textContent = 'Please try again';
                    resultIcon.innerHTML = '<i class="material-icons red-text">error</i>';
                }
            }
        });
    }

    // WebSocket handling
    if (typeof socket !== 'undefined') {
        socket.onmessage = function(event) {
            var result = JSON.parse(event.data);
            console.log('Result:', result);
            var confidence = Math.min(result.confidence, 100);
            confidence = confidence.toFixed(2);
            const realTimeResult = document.getElementById('realTimeResult');
            if (realTimeResult) {
                realTimeResult.innerHTML = "Detected: " + result.prediction + ".<br>Confidence: " + confidence + "%";
            }
        };
    }
});

document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------
    // 1. Toast Notification Helper
    // ----------------------------------------------------
    const toastEl = document.getElementById('liveToast');
    const toastMessage = document.getElementById('toastMessage');
    let toast;
    if (toastEl) {
        toast = new bootstrap.Toast(toastEl);
    }

    function showToast(message, type = 'success') {
        if (!toastEl) return alert(message);
        toastMessage.textContent = message;
        toastEl.className = `toast align-items-center text-bg-${type} border-0`;
        toast.show();
    }

    // ----------------------------------------------------
    // 2. Text-to-Speech (Hindi) Helper
    // ----------------------------------------------------
    function speakHindi(text) {
        if ('speechSynthesis' in window) {
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'hi-IN';
            utterance.rate = 0.9; // Slightly slower for clarity
            
            // Try to find a Hindi voice specifically
            const voices = window.speechSynthesis.getVoices();
            const hindiVoice = voices.find(v => v.lang === 'hi-IN' || v.lang === 'hi_IN');
            if (hindiVoice) {
                utterance.voice = hindiVoice;
            }
            
            window.speechSynthesis.speak(utterance);
        } else {
            console.warn("Text-to-speech not supported in this browser.");
        }
    }

    // Initialize voices immediately
    if ('speechSynthesis' in window) {
        window.speechSynthesis.getVoices();
    }

    // ----------------------------------------------------
    // 3. Form Submission & API Calls
    // ----------------------------------------------------
    const predictionForm = document.getElementById('predictionForm');
    const resultContainer = document.getElementById('resultContainer');
    const predictionResult = document.getElementById('predictionResult');
    let resultChartInstance = null; // Store chart instance to destroy it later

    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Gather form data
            const formData = new FormData(predictionForm);
            const data = Object.fromEntries(formData.entries());
            const endpoint = predictionForm.getAttribute('data-endpoint');
            
            // Change button state
            const submitBtn = predictionForm.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> कृपया प्रतीक्षा करें... (Loading)';
            submitBtn.disabled = true;

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    predictionResult.textContent = result.prediction;
                    resultContainer.classList.remove('d-none');
                    resultContainer.scrollIntoView({ behavior: 'smooth' });
                    
                    showToast("अनुमान सफलता पूर्वक प्राप्त हुआ!");
                    speakHindi("आपका परिणाम है: " + result.prediction);
                    
                    // Render Charts if applicable
                    renderCharts(endpoint, data, result.prediction);

                } else {
                    showToast('त्रुटि: ' + (result.error || 'Unknown error occurred.'), 'danger');
                }
            } catch (error) {
                console.error('Error:', error);
                showToast('नेटवर्क एरर। कृपया फिर से प्रयास करें।', 'danger');
            } finally {
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }

    // ----------------------------------------------------
    // 4. Chart Rendering Logic
    // ----------------------------------------------------
    function renderCharts(endpoint, inputData, predictionText) {
        const ctx = document.getElementById('resultChart');
        if (!ctx) return; // No canvas on this page
        
        if (resultChartInstance) {
            resultChartInstance.destroy();
        }

        if (endpoint.includes('yield')) {
            // Yield Bar Chart: Input variables comparison (normalized just for visual)
            resultChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Area (ha)', 'Rainfall (mm)', 'Temperature (°C)', 'Fertilizer (kg)'],
                    datasets: [{
                        label: 'खेत के पैरामीटर्स (Farm Parameters)',
                        data: [inputData.area, inputData.rainfall, inputData.temperature, inputData.fertilizer],
                        backgroundColor: ['#4CAF50', '#2196F3', '#FF9800', '#795548']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        } else if (endpoint.includes('recommend')) {
            // Recommendation Radar Chart: N, P, K, pH profile
            resultChartInstance = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'pH Level (x10)', 'Rainfall (mm/10)'],
                    datasets: [{
                        label: 'मिट्टी की प्रोफ़ाइल (Soil Profile)',
                        data: [inputData.n, inputData.p, inputData.k, parseFloat(inputData.ph)*10, parseFloat(inputData.rainfall)/10],
                        backgroundColor: 'rgba(0, 188, 212, 0.2)',
                        borderColor: 'rgba(0, 188, 212, 1)',
                        pointBackgroundColor: 'rgba(0, 188, 212, 1)'
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }
    }

    // ----------------------------------------------------
    // 5. PDF Export Logic
    // ----------------------------------------------------
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener('click', () => {
            const element = document.getElementById('reportContent');
            if (!element) return;
            
            showToast("PDF रिपोर्ट तैयार हो रही है... (Generating PDF...)", "info");
            
            const opt = {
              margin:       1,
              filename:     'Kisan_Sahayak_Report.pdf',
              image:        { type: 'jpeg', quality: 0.98 },
              html2canvas:  { scale: 2 },
              jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
            };
            
            html2pdf().set(opt).from(element).save().then(() => {
                showToast("PDF सफलतापूर्वक डाउनलोड हो गई!");
            });
        });
    }

    // ----------------------------------------------------
    // 6. Voice Input (Speech-to-Text)
    // ----------------------------------------------------
    const voiceBtn = document.getElementById('voiceBtn');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const voiceTranscript = document.getElementById('voiceTranscript');
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    if (voiceBtn) {
        voiceBtn.addEventListener('click', async () => {
            if (isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                voiceBtn.innerHTML = '<i class="fas fa-microphone me-2"></i> बोलकर भरें (Voice Input)';
                voiceBtn.classList.remove('btn-secondary', 'mic-animation');
                voiceBtn.classList.add('btn-danger');
                recordingIndicator.classList.add('d-none');
                voiceTranscript.textContent = "प्रोसेसिंग... (Processing audio...)";
            } else {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.addEventListener("dataavailable", event => {
                        audioChunks.push(event.data);
                    });

                    mediaRecorder.addEventListener("stop", async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        await processVoice(audioBlob);
                        stream.getTracks().forEach(track => track.stop());
                    });

                    mediaRecorder.start();
                    isRecording = true;
                    voiceBtn.innerHTML = '<i class="fas fa-stop-circle me-2"></i> रोकें (Stop)';
                    voiceBtn.classList.remove('btn-danger');
                    voiceBtn.classList.add('btn-secondary', 'mic-animation');
                    recordingIndicator.classList.remove('d-none');
                    voiceTranscript.textContent = "";

                } catch (err) {
                    console.error("Microphone access denied:", err);
                    showToast("माइक का एक्सेस नहीं मिला। (Microphone access denied.)", "danger");
                }
            }
        });
    }

    async function processVoice(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob);

        try {
            const response = await fetch('/api/voice', { method: 'POST', body: formData });
            const result = await response.json();

            if (response.ok) {
                voiceTranscript.textContent = "आपने कहा: " + result.text;
                fillFormFromText(result.text);
                showToast("आवाज़ को सफलतापूर्वक समझ लिया गया।", "success");
            } else {
                voiceTranscript.textContent = "त्रुटि: " + result.error;
                voiceTranscript.classList.replace('text-primary', 'text-danger');
                showToast(result.error, "danger");
            }
        } catch (error) {
            console.error('Voice processing error:', error);
            voiceTranscript.textContent = "सर्वर से कनेक्ट नहीं हो सका।";
            voiceTranscript.classList.replace('text-primary', 'text-danger');
        }
    }

    function fillFormFromText(text) {
        const numbers = text.match(/\d+(\.\d+)?/g);
        if (!numbers || numbers.length === 0) {
            showToast("हम आपकी आवाज़ से नंबर नहीं पहचान पाए। (Could not extract numbers.)", "warning");
            return;
        }

        const inputs = Array.from(predictionForm.querySelectorAll('input[type="number"]'));
        for (let i = 0; i < Math.min(numbers.length, inputs.length); i++) {
            inputs[i].value = numbers[i];
            inputs[i].classList.add('is-valid');
            setTimeout(() => inputs[i].classList.remove('is-valid'), 2000);
        }
    }

    // ----------------------------------------------------
    // 7. Weather API Auto-fill (Open-Meteo)
    // ----------------------------------------------------
    const weatherBtn = document.getElementById('weatherBtn');
    if (weatherBtn) {
        weatherBtn.addEventListener('click', () => {
            if ("geolocation" in navigator) {
                weatherBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> मौसम ला रहे हैं...';
                weatherBtn.disabled = true;

                navigator.geolocation.getCurrentPosition(async (position) => {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    
                    try {
                        // Open-Meteo API
                        const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,precipitation`;
                        const res = await fetch(url);
                        const weatherData = await res.json();
                        
                        if (weatherData && weatherData.current) {
                            const temp = weatherData.current.temperature_2m;
                            const humidity = weatherData.current.relative_humidity_2m;
                            const rain = weatherData.current.precipitation || 0;
                            
                            // Auto-fill fields if they exist on the current page
                            const tempInput = document.getElementById('temperature');
                            const humInput = document.getElementById('humidity');
                            const rainInput = document.getElementById('rainfall');
                            
                            if (tempInput) { tempInput.value = temp; flashInput(tempInput); }
                            if (humInput) { humInput.value = humidity; flashInput(humInput); }
                            if (rainInput) { rainInput.value = rain > 0 ? rain : 100; flashInput(rainInput); } // Provide fallback rain if 0
                            
                            showToast("मौसम का डाटा सफलतापूर्वक भर दिया गया है!", "success");
                        }
                    } catch (err) {
                        console.error(err);
                        showToast("मौसम डाटा लाने में विफल।", "danger");
                    } finally {
                        weatherBtn.innerHTML = '<i class="fas fa-cloud-sun me-2"></i> मौसम भरें (Auto Weather)';
                        weatherBtn.disabled = false;
                    }
                }, (error) => {
                    showToast("लोकेशन एक्सेस नहीं मिला। (Location access denied)", "danger");
                    weatherBtn.innerHTML = '<i class="fas fa-cloud-sun me-2"></i> मौसम भरें (Auto Weather)';
                    weatherBtn.disabled = false;
                });
            } else {
                showToast("आपका ब्राउज़र लोकेशन सपोर्ट नहीं करता।", "warning");
            }
        });
    }

    function flashInput(el) {
        el.classList.add('is-valid');
        setTimeout(() => el.classList.remove('is-valid'), 2000);
    }
});

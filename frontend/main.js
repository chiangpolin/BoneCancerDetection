const API_URL =
  'https://bone-cancer-detection-232107405208.us-central1.run.app';
const imageInput = document.getElementById('imageInput');
const predictBtn = document.getElementById('predictBtn');
const preview = document.getElementById('preview');
const predictionText = document.getElementById('predictionText');
const percentage = document.getElementById('percentage');

let selectedFile = null;

// preview image
imageInput.addEventListener('change', (e) => {
  selectedFile = e.target.files[0];
  if (!selectedFile) return;

  const reader = new FileReader();
  reader.onload = function (event) {
    preview.src = event.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(selectedFile);
});

// call API
predictBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    alert('Please select an image first!');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);

  predictionText.textContent = 'Predicting...';

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    predictionText.textContent = `${data.prediction}`;
    percentage.textContent = `Confidence: ${
      data.confidence.toFixed(2) * 100
    } %`;
  } catch (error) {
    predictionText.textContent = 'Error predicting image.';
    console.error(error);
  }
});

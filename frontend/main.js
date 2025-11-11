const imageInput = document.getElementById('imageInput');
const predictBtn = document.getElementById('predictBtn');
const preview = document.getElementById('preview');
const predictionText = document.getElementById('predictionText');

let selectedFile = null;

// 顯示圖片預覽
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

// 呼叫後端 API 進行預測
predictBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    alert('Please select an image first!');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);

  predictionText.textContent = 'Predicting...';

  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    predictionText.textContent = `Prediction: ${data.prediction}, Confidence: ${data.confidence}`;
  } catch (error) {
    predictionText.textContent = 'Error predicting image.';
    console.error(error);
  }
});

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import uvicorn


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Can also use ["*"] to allow all (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],    # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],    # Allow all headers
)

# load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# preprocess image
tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

@app.get("/")
def read_root():
    return {"message": "hi!"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = tfms(img).unsqueeze(0)

    with torch.no_grad():
        outputs = F.softmax(model(x), dim=1)
        conf, pred = torch.max(outputs, 1)

    label = "cancer" if pred.item() == 0 else "normal"

    return JSONResponse({
        "prediction": label,
        "confidence": float(conf.item())
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
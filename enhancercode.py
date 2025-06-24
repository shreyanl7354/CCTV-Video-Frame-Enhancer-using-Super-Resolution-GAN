import cv2
from realesrgan import RealESRGAN
from PIL import Image
import numpy as np

# Load your Real-ESRGAN model
model = RealESRGAN('cuda')  # or 'cpu' if you don't have a GPU
model.load_weights('RealESRGAN_x4.pth')  # ensure this path is correct

# Load video
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('enhanced_output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 4, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 4))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame to PIL Image for Real-ESRGAN
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Super-resolve
    sr_image = model.predict(pil_frame)
    # Convert back to OpenCV format
    sr_frame = cv2.cvtColor(np.array(sr_image), cv2.COLOR_RGB2BGR)
    out.write(sr_frame)

cap.release()
out.release()
print("Enhancement complete!")
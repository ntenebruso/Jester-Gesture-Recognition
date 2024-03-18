import cv2
import numpy as np
from tensorflow.keras.models import load_model
from threading import Thread

model = load_model("model_final.keras")

cap = cv2.VideoCapture(0)
frames = []
cooldown = 0

classes = np.load("encoder_classes.npy", allow_pickle=True)[0]

def predict_model(model_input):
  output = model.predict(model_input[None,:])[0]
  print(output, np.argmax(output), classes[np.argmax(output)], flush=True)
  

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't retrieve stream.")
        break

    inp = frame
    inp = cv2.resize(inp, (56, 56))

    frames.append(inp)

    if len(frames) > 36:
        frames.pop(0)

    if len(frames) == 36 and cooldown == 0:
        model_input = np.asarray(frames)[None,:]
        print(model_input.shape)
        run_thread = Thread(target=predict_model, args=(model_input))
        run_thread.start()
        cooldown = 36
    
    if cooldown > 0:
        cooldown -= 1
    
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

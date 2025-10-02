from ultralytics import SAM

model = SAM("sam2.1_t.pt")
model.info()

results = model("/data/vedio/3.mp4", device="cuda:0", show=True, save=True)

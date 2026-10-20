import numpy as np
from PIL import Image

# 1. Create a dummy skin image for Tab 1
img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save("test_skin_lesion.jpg")
print("Saved: test_skin_lesion.jpg")

# 2. Create 3 EMG signal test CSVs for Tab 2
np.random.seed(42)

# Healthy signal (normal baseline)
healthy = np.random.normal(loc=0.0, scale=0.5, size=(1000,))
np.savetxt("healthy_test.csv", healthy, delimiter=",")

# Neuropathy signal (high-amplitude spikes)
neuropathy = np.random.normal(loc=0.0, scale=0.8, size=(1000,))
neuropathy[::50] += np.random.uniform(2.0, 4.0, size=20)
np.savetxt("neuropathy_test.csv", neuropathy, delimiter=",")

# Myopathy signal (low-amplitude high frequency)
myopathy = np.random.normal(loc=0.0, scale=0.2, size=(1000,))
np.savetxt("myopathy_test.csv", myopathy, delimiter=",")

print("Saved: healthy_test.csv, neuropathy_test.csv, myopathy_test.csv")
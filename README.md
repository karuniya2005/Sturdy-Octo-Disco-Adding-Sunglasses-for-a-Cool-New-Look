# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

## Program:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Face Image
faceImage = cv2.imread('ka.JPG')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
```

<img width="565" height="521" alt="image" src="https://github.com/user-attachments/assets/c85f1d5b-c909-4fc4-8ebb-fe90d8817960" />


```
#resized_faceImage.shape
faceImage.shape
```



<img width="170" height="38" alt="image" src="https://github.com/user-attachments/assets/2d1d06c1-0145-4dc5-82c9-956402098672" />

```
glassPNG = cv2.imread('sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
```


<img width="697" height="347" alt="image" src="https://github.com/user-attachments/assets/f87e6514-a004-4a93-9ceb-b5c3c19c0662" />
<img width="286" height="39" alt="image" src="https://github.com/user-attachments/assets/70c878ed-b76e-4fd2-b57c-243a33f46d81" />


```
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```



<img width="845" height="155" alt="image" src="https://github.com/user-attachments/assets/cf30938d-d94f-43c3-853d-066289b720d0" />


```
import cv2
import numpy as np
import matplotlib.pyplot as plt

faceWithGlassesNaive = faceImage.copy()

# Target position
x, y = 118, 200

# Desired size
target_w, target_h = 160, 60

# Resize glasses
glassResized = cv2.resize(glassBGR, (target_w, target_h))

# Get image dimensions
h, w = faceWithGlassesNaive.shape[:2]

# Clip if overlay goes out of bounds
end_x = min(x + target_w, w)
end_y = min(y + target_h, h)

# Adjust overlay size to match clipped ROI
glassResized = glassResized[:end_y - y, :end_x - x]

# Overlay
faceWithGlassesNaive[y:end_y, x:end_x] = glassResized

plt.imshow(faceWithGlassesNaive[..., ::-1])
plt.axis("off")
plt.show()
```


<img width="377" height="434" alt="image" src="https://github.com/user-attachments/assets/d9eace0e-7b52-4698-aabc-503424bd288f" />


```
# Assuming glassPNG has alpha channel
glassAlpha = glassPNG[..., 3] / 255.0
glassBGR = glassPNG[..., :3]

# Resize
glassBGR = cv2.resize(glassBGR, (target_w, target_h))
glassAlpha = cv2.resize(glassAlpha, (target_w, target_h))

# Extract eye region from face
eyeRoi = faceImage[y:y+target_h, x:x+target_w].copy()

# Masked eye and glasses
maskedEye = eyeRoi * (1 - glassAlpha[..., np.newaxis])
maskedGlass = glassBGR * glassAlpha[..., np.newaxis]
eyeRoiFinal = maskedEye + maskedGlass

# Convert to uint8 for display
maskedEye_disp = np.clip(maskedEye.astype(np.uint8), 0, 255)
maskedGlass_disp = np.clip(maskedGlass.astype(np.uint8), 0, 255)
eyeRoiFinal_disp = np.clip(eyeRoiFinal.astype(np.uint8), 0, 255)

# Display 3-panel intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131)
plt.imshow(maskedEye_disp[..., ::-1])
plt.title("Masked Eye Region")
plt.axis("off")

plt.subplot(132)
plt.imshow(maskedGlass_disp[..., ::-1])
plt.title("Masked Sunglass Region")
plt.axis("off")

plt.subplot(133)
plt.imshow(eyeRoiFinal_disp[..., ::-1])
plt.title("Augmented Eye and Sunglass")
plt.axis("off")
plt.show()
```


<img width="826" height="134" alt="image" src="https://github.com/user-attachments/assets/bd8a3f3a-f5f2-45a7-b7a8-f06b99bbd978" />

```
# Create a copy of the original face
faceWithGlasses = faceImage.copy()

# Apply alpha blending of sunglasses (example)
x, y = 102, 195
target_w, target_h = 192, 80

# Image dimensions
h, w = faceWithGlasses.shape[:2]

# Clip so sunglasses don't exceed boundaries
end_x = min(x + target_w, w)
end_y = min(y + target_h, h)

# Adjust width/height based on clipping
overlay_w = end_x - x
overlay_h = end_y - y

# Resize glasses to fit adjusted size
glassBGR_resized = cv2.resize(glassBGR, (overlay_w, overlay_h))
glassAlpha_resized = cv2.resize(glassPNG[..., 3] / 255.0, (overlay_w, overlay_h))

# Extract ROI of face
roi = faceWithGlasses[y:end_y, x:end_x]

# Alpha blending
for c in range(3):
    roi[..., c] = roi[..., c] * (1 - glassAlpha_resized) + glassBGR_resized[..., c] * glassAlpha_resized

# Put ROI back into image
faceWithGlasses[y:end_y, x:end_x] = roi

# Plot
plt.figure(figsize=[15, 8])
plt.subplot(1, 2, 1)
plt.imshow(faceImage[..., ::-1])
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(faceWithGlasses[..., ::-1])
plt.title("Face with Sunglasses")
plt.axis("off")

plt.tight_layout()
plt.show()
```

<img width="776" height="499" alt="image" src="https://github.com/user-attachments/assets/901b169d-3f20-45f3-bd0c-3ecd444e2ce1" />










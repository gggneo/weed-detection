# weed-detection
## OneAPI Weed Detectection Heckathon

Requirements:
1. Tensorflow >= 1.15.0
2. Opencv
3. tqdm
4. pillow

### Openvino Toolkit
OneAPI -- GAL, MKL

### steps
### 1. To run TF model command:
python image_demo.py

### 2. To run IR model command :
python inference.py

### Inference Demo:
![Inference_time](https://user-images.githubusercontent.com/43141616/221826012-7b864755-419a-4285-ac16-9187b6ce9980.png)</br>

### Output from TF model:
![TF_Model_Output](https://user-images.githubusercontent.com/43141616/221826427-952c840a-ace1-4539-8ad7-ff90bcfe9e73.png)</br>

### Output From IR model:
![output_IR](https://user-images.githubusercontent.com/43141616/221826624-45790b4c-5f95-47a9-b3c1-e4c515993344.jpg)

#### NOTE:
You can run Inference on your dataset by changing as per your file location in the image_demo.py for TF model and inference.py for IR model.

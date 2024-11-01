# Plant Seedling Classification

<p align="left"> 
</a>   <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> 
</a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

### Overview
This project aims to classify different species of plants at the seedling stage using image recognition techniques. Early identification of plant species can significantly improve agricultural practices by aiding in efficient crop management and invasive species detection.

Using deep learning and computer vision, this project leverages a Convolutional Neural Network (CNN) model to accurately classify 12 plant species. The model is trained on a labeled dataset of seedling images, employing data preprocessing, augmentation, and a CNN-based architecture to achieve high accuracy in predicting plant species.
# Dataset
The Plant Seedlings Dataset from Kaggle is used for training and testing the model. It contains labeled images of seedlings across 12 species:
- Black-grass
- Charlock
- Cleavers
- Common Chickweed
- Common Wheat
- Fat Hen
- Loose Silky-bent
- Maize
- Scentless Mayweed
- Shepherd’s Purse
- Small-flowered
- Cranesbill
- Sugar Beet
___
# Project Structure
The project is organized as follows:
```
plant_seedling_classification/
├── data/
│   ├── train/                 # Training images categorized by folders for each species
│   └── test/                  # Test images for evaluation
├── src/
│   ├── data_preprocessing.py  # Image resizing, normalization, and augmentation code
│   ├── model.py               # Model architecture and training code
│   ├── evaluate.py            # Evaluation metrics
│   └── inference.py           # Code for predicting on new images
├── notebooks/                 # Jupyter notebooks for EDA, training, and experiments
├── README.md                  # Project overview
├── requirements.txt           # Dependencies
└── LICENSE                    # License information
```
## Technologies and Libraries Used

- **Python** - Core programming language for the project.
- **TensorFlow/Keras** - Deep learning framework for building and training the model.
- **OpenCV** - Image preprocessing library.
- **Pandas** - Data manipulation.
- **NumPy** - Numerical computations.
- **Matplotlib/Seaborn** - Data visualization.
- **scikit-learn** - Evaluation metrics and preprocessing.
- **Jupyter Notebooks** - EDA and experimentation environment.
- **Pickle** - Saving/loading models.
- **Git/GitHub** - Version control and project hosting.
- **ImageDataGenerator (Keras)** - Augmentation tool for training images.
- **Docker** - Containerization for deployment.
## The model i created for seedling classification
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(12, activation='softmax')])
```
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15 
)
```
## Classificator demonstration in Streamlit
### Screenshots
![Alt Text](/Screenshot%202024-11-01%20134733.png)
## Results
The final model achieved an accuracy of 87% on the test set, demonstrating robust performance across various species.
## Contributing
Contributions to improve model performance, code structure, or documentation are welcome. Please see the CONTRIBUTING.md for guidelines.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

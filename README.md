Introduction:
	The aim of the Plant Disease Detection from Leaf Images project is to develop a system that can automatically identify and classify diseases affecting plants based on images of their leaves. 
 I am building four models: CNN, VGG16, ResNet, and MobileNet, and finally deploying them in a Streamlit application.

Data Loading:
	Downloading the data from Kaggle, loading it, and splitting the dataset into training, testing, and validation sets, while saving the datasets. The training set will consist of 70% of the data,
 and the testing and validation sets will each comprise 15% of the data.

Building the Model:
	Loading the training, testing, and validation data, and applying rescaling, rotation, horizontal and vertical shifting, and zooming in and out on the images. Then, I built the models: first, 
 the CNN with early stopping, achieving an accuracy of 0.9003 and a loss of 0.2894; next, the VGG16 model with an accuracy of 0.6780 and a loss of 0.9793; then, the ResNet model with an accuracy of 0.4034 and a loss of 1.7980; and finally, the MobileNet model with an accuracy of 0.9042 and a loss of 0.2671. I saved all four models for future use, tested them on the test dataset, and plotted the training accuracy, validation accuracy, training loss, and validation loss.

Streamlit:
	On the Streamlit page, we want to upload the image in the center, and in the sidebar, we want to upload the saved model. The model must be a .pkl file, trained on a dataset with 15 classes, 
 and the input size should be 150x150 pixels. When we click the "Run Detection" button, it will display the result of the image prediction, indicating whether the leaf is healthy or has a disease. If the leaf has any diseases, it will also provide recommendations to maintain the leaf's health.

Conclusion:
	The Plant Disease Detection from Leaf Images project effectively utilizes deep learning techniques to identify and classify plant diseases through leaf images. By developing and evaluating 
 four models—CNN, VGG16, ResNet, and MobileNet. I achieved notable accuracy, with MobileNet performing the best at 0.9042. The project involved comprehensive data preparation and augmentation, 
 culminating in a user-friendly Streamlit application that allows users to upload leaf images and receive instant predictions on leaf health, along with recommendations for disease management. 
 This initiative demonstrates the significant potential of machine learning in enhancing agricultural practices and supporting plant health monitoring.

DATASET: https://www.kaggle.com/datasets/emmarex/plantdisease
VIDEO DEMO: https://drive.google.com/drive/folders/1xNTkwNfqU4iNT_ARf_maM1f-vY3FH8UC?usp=sharing

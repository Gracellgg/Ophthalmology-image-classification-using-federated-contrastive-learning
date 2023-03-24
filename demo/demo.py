import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QFont, QIcon

from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import vgg11, resnet18
import torchvision.models as models
import torchvision.transforms as transforms

# import simclr model from simCLR file
from simCLR.simclr import SimCLR
from torchvision.transforms import RandomApply, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, Grayscale



# Load the pre-trained PyTorch model
# model = torch.load('model.pth')
#model = vgg11(pretrained=True)

base_encoder = models.resnet18(pretrained=True)
base_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_features = base_encoder.fc.in_features
base_encoder.fc = nn.Linear(num_features, out_features=512)
model = SimCLR(base_encoder)
model.load_state_dict(torch.load('simCLR_model_50.pt'))

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
transform = transforms.Compose([
    RandomApply([RandomResizedCrop(size=(128, 128), scale=(0.2, 1.0)), RandomHorizontalFlip()], p=0.5),
    RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])
# Read the class names
with open('labels.txt', 'r') as f:
    classes = eval(f.read())


# Create a function that takes an image as input and returns the predicted class and confidence score
def predict(image):
    # Preprocess the image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    # Pass the image through the model
    with torch.no_grad():
        output = model(image_tensor)
    # Get the top 3 predicted class and confidence score
    top3_confidence_score, top3_predicted_class = output[0].topk(3)
    top3_predicted_classname = [classes[i.item()].split(', ')[0] for i in top3_predicted_class]
    top3_confidence_score = [round(i.item(), 2) for i in top3_confidence_score]
    return top3_predicted_classname, top3_confidence_score


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ImageNet Classification')
        self.setGeometry(100, 100, 850, 500)
        self.setWindowIcon(QIcon('imagenet.jpg'))
        # Set the background color of the GUI to white.
        self.setStyleSheet("background-color: white;")

        # Create a label for displaying the title of the GUI
        title_label = QLabel('ImageNet Classification', self)
        title_label.move(20, 0)
        title_label.resize(820, 100)
        title_label.setFont(QFont("Arial", 18))
        title_label.setAlignment(Qt.AlignCenter)

        # Create a label for displaying the background color of the uploaded image
        self.background_label = QLabel(self)
        self.background_label.move(20, 100)
        self.background_label.resize(420, 300)
        self.background_label.setStyleSheet('''
            QLabel {
                background-color: #f0f0f0;
                border-radius: 10px;
            }
        ''')
        self.background_label.setAlignment(Qt.AlignCenter)

        # Create a label for displaying the output
        self.output_label = QLabel(self)
        self.output_label.move(480, 100)
        self.output_label.resize(350, 250)
        self.output_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.output_label.setStyleSheet('''
            QLabel {
                color: #000000;
                border-color: #b5b5b5;
                border-width: 1px;
                border-style: solid;
                border-radius: 10px;
            }
        ''')

        # Create a class label for displaying the predicted class
        self.class_label = QLabel(self)
        self.class_label.move(480, 100)
        self.class_label.resize(350, 80)
        self.class_label.setFont(QFont("Arial", 16))
        self.class_label.raise_()
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setAttribute(Qt.WA_TranslucentBackground)

        # Create a left label for displaying the top3 class names
        self.left_label = QLabel(self)
        self.left_label.move(490, 200)
        self.left_label.resize(165, 215)
        self.left_label.setFont(QFont("Arial", 12))
        self.left_label.setAlignment(Qt.AlignLeft)
        self.left_label.setAttribute(Qt.WA_TranslucentBackground)
        self.left_label.setStyleSheet('''
            QLabel {
                color: #505050;
            }
        ''')

        # Create a right label for displaying the top3 confidence score
        self.right_label = QLabel(self)
        self.right_label.move(655, 200)
        self.right_label.resize(165, 215)
        self.right_label.setFont(QFont("Arial", 12))
        self.right_label.setAlignment(Qt.AlignRight)
        self.right_label.setAttribute(Qt.WA_TranslucentBackground)
        self.right_label.setStyleSheet('''
            QLabel {
                color: #505050;
            }
        ''')

        # Create a button for uploading images
        upload_button = QPushButton('Submit', self)
        upload_button.move(240, 420)
        upload_button.resize(200, 50)
        upload_button.setFont(QFont("Arial", 12, QFont.Bold))
        upload_button.setStyleSheet('''
            QPushButton {
                background-color: #FFE5B4;
                border-style: outset;
                border-width: 1px;
                border-color: #FFA500;
                border-radius: 10px;
                color: #FFA500;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #FFDAB9;
            }
        ''')
        upload_button.clicked.connect(self.showDialog)

        # Create a button for canceling the upload
        cancel_button = QPushButton('Cancel', self)
        cancel_button.move(20, 420)
        cancel_button.resize(200, 50)
        cancel_button.setFont(QFont("Arial", 12, QFont.Bold))
        cancel_button.setStyleSheet('''
            QPushButton {
                background-color: #f7f7f7;
                border: 1px solid #b5b5b5;
                border-radius: 10px;
                color: #555555;
                padding: 6px 12px;
            }

            QPushButton:hover {
                background-color: #555555;
                color: #f7f7f7;
            }
        ''')
        cancel_button.clicked.connect(self.cancel)

    def showDialog(self):
        # Open a file dialog box for users to select an image file from their computer.
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')

        if fname[0]:
            # Display the uploaded image and predicted class/ confidence score in our GUI label.
            pixmap = QPixmap(fname[0])
            pixmap_resized = pixmap.scaled(420, 300, Qt.KeepAspectRatio)
            self.background_label.setPixmap(pixmap_resized)
            # Load the selected image using PIL.Image, and pass it to the predict function.
            img = Image.open(fname[0])
            top3_predicted_classname, top3_confidence_score = predict(img)
            self.showOutput(top3_predicted_classname, top3_confidence_score)

    def showOutput(self, top3_class, top3_confidence):
        text_label = f'{top3_class[0]}'
        self.class_label.setText(text_label)
        class_label = ""
        confidence_label = ""
        for i in range(len(top3_class)):
            class_label += f'{top3_class[i]}\n\n'
            confidence_label += f'{top3_confidence[i]}\n\n'
        self.left_label.setText(class_label)
        self.right_label.setText(confidence_label)

    def cancel(self):
        self.background_label.clear()
        self.class_label.clear()
        self.left_label.clear()
        self.right_label.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())


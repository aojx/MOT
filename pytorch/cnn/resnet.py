import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchvision import transforms, datasets
import os
import numpy as np
from datetime import datetime
import cv2
from gradcam import GradCAM

########################################################################################
# 환경변수
########################################################################################
IS_RUN_LEARNING = True   # 신규 학습 수행 여부. False 를 선택하면 기존에 생성한 가장 최근 모델 파일(*.pth)을 GradCAM 에 사용
EPOCHS = 10
INPUT_IMG_SIZE = 384
DATA_DIR = "./filtered"   # 데이터 경로 설정. 서브 디렉토리명을 클래스 분류 기준으로 사용함
########################################################################################

# GPU 사용 가능 여부 확인
#device = "cuda" if torch.cuda.is_available() else "cpu"    # for NVIDA CUDA
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')  # for Apple Silicon

print(f"device 정보: {device}")

# 이미지 전처리
transform = transforms.Compose([
    #transforms.Resize((128, 128)),  # 이미지 크기 조절
    transforms.ToTensor(),         # 이미지를 Tensor로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지 정규화
])

# 데이터셋 생성
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print(dataset)

# 데이터셋의 클래스 확인
classes = dataset.classes
num_classes = len(classes)
print(f"Classes: {classes}")
print(f"Number of classes: {num_classes}")

# 데이터 분할: 학습 데이터셋과 테스트 데이터셋 분리
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 학습 데이터셋의 파일명, 레이블 및 이미지 그리기
def show_filenames_and_labels(dataset, dataset_indices):
    for idx in dataset_indices:
        filename = dataset.dataset.imgs[idx][0].split('/')[-1]  # 파일명 가져오기
        label = dataset.dataset.targets[idx]                   # 레이블 가져오기
        classname = dataset.dataset.classes[label]             # 레이블을 클래스 이름으로 변환
        print(f"Filename: {filename}, Label: {classname}")

# 학습 데이터셋의 파일명과 레이블 출력
show_filenames_and_labels(train_dataset, train_dataset.indices)

############################################
# ResNet
############################################
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=2, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

############################################
# End of ResNet
############################################

# 데이터 로더 생성
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=42, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=42, shuffle=True)

model = resnet50().to(device)
x = torch.randn(3 , 3, INPUT_IMG_SIZE, INPUT_IMG_SIZE).to(device)
output = model(x)
print(output.size())
summary(model, (1, 3, INPUT_IMG_SIZE, INPUT_IMG_SIZE), depth=3)

n_params = sum(p.numel() for p in model.parameters())

print("\n===== Model Architecture =====")
print(model, "\n")

print("\n===== Model Parameters =====")
print(" - {}".format(n_params), "\n\n")

# 손실 함수 및 최적화 기법 정의
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)

# 학습 및 테스트 함수 정의
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def test_per_image(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(inputs)):
                filename = test_loader.dataset.dataset.imgs[i][0].split('/')[-1]
                label = test_loader.dataset.dataset.classes[labels[i]]
                prediction = test_loader.dataset.dataset.classes[predicted[i]]
                print(f"Filename: {filename}, Label: {label}, Prediction: {prediction}")

# 학습
if IS_RUN_LEARNING :
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        test_loss, test_acc = test(model, test_loader, criterion)
        print(f"Epoch {epoch}/{EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        # 테스트할 때 파일별로 결과 출력
        test_per_image(model, test_loader, criterion)

    # 모델 저장
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    PATH = f"resnet50_model_{current_time}.pth"
    torch.save(model, PATH)


############################################
# GradCAM
############################################
from PIL import Image

# defines two global scope variables to store our gradients and activations
gradients = {}
activations = {}

def forward_hook(module, input, output):
    global activations
    activations['value'] = output
    print(f'Forward hook - activations: {activations["value"].shape}')

def backward_hook(module, grad_input, grad_output):
    global gradients
    if isinstance(grad_output, (list, tuple)):
        gradients['value'] = grad_output[0]
    else:
        gradients['value'] = grad_output
    print(f'Backward hook - gradients: {gradients["value"].shape}')

# Grad-CAM을 위한 함수 정의
def grad_cam(model, img_path):
    global activations, gradients
    gradients = {}
    activations = {}

    # 이미지 전처리
    preprocess = transforms.Compose([
        #transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    model.eval()

    # 마지막 합성곱 레이어에 훅 등록
    target_layer = model.conv5_x[2].residual_function[6]
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # 모델 예측 및 역전파
    output = model(img_tensor.to(device))
    print(f'Model output: {output}')
    target_class = output.argmax().item()  # 예측된 클래스를 target class로 사용
    print(f'Target class: {target_class}')
    model.zero_grad()
    output[:, target_class].backward()

    # Grad-CAM 계산
    grads = gradients['value'].cpu().data.numpy()[0]
    activations = activations['value'].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (INPUT_IMG_SIZE, INPUT_IMG_SIZE))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    original_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) / 255
    grad_cam_img = heatmap + original_img
    grad_cam_img = grad_cam_img / np.max(grad_cam_img)

    handle_forward.remove()
    handle_backward.remove()

    return (grad_cam_img * 255).astype(np.uint8), target_class

############################################
# End of GradCAM
############################################

def get_latest_model_path(directory=".", prefix="resnet50_model_", suffix=".pth"):
    # 디렉토리 내의 파일 목록을 가져옵니다.
    files = os.listdir(directory)
    
    # 지정된 prefix와 suffix를 가진 파일들만 필터링합니다.
    model_files = [f for f in files if f.startswith(prefix) and f.endswith(suffix)]
    
    # 모델 파일이 없는 경우 예외 처리
    if not model_files:
        raise FileNotFoundError(f"No model files found with prefix '{prefix}' and suffix '{suffix}' in directory '{directory}'")
    
    # 파일명을 기준으로 가장 최신 파일을 찾습니다.
    latest_model_file = max(model_files, key=lambda x: x[len(prefix):-len(suffix)])
    
    return os.path.join(directory, latest_model_file)

# 이미지 경로 설정
resized_dir = "./resized"
filtered_dir = "./filtered"
finalized_dir = "./finalized"

# 디렉토리 생성
os.makedirs(finalized_dir, exist_ok=True)

MODEL_PATH = get_latest_model_path()
# 저장된 모델의 state_dict를 불러오기
res_model = torch.load(MODEL_PATH)
res_model.eval()

# Grad-CAM 생성 및 저장
for category in ['OK', 'NG']:
    resized_category_dir = os.path.join(resized_dir, category)
    filtered_category_dir = os.path.join(filtered_dir, category)
    finalized_category_dir = os.path.join(finalized_dir, category)

    # 카테고리별 디렉토리 생성
    os.makedirs(finalized_category_dir, exist_ok=True)

    for filename in os.listdir(filtered_category_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(filtered_category_dir, filename)
            print(img_path)
            grad_cam_img, target_class = grad_cam(res_model, img_path)

            # 원본 이미지, 필터링된 이미지, Grad-CAM 이미지 로드
            original_img_path = os.path.join(resized_category_dir, filename)
            filtered_img_path = os.path.join(filtered_category_dir, filename)

            if os.path.exists(original_img_path) and os.path.exists(filtered_img_path):
                original_img = cv2.imread(original_img_path)
                filtered_img = cv2.imread(filtered_img_path)

                # Ensure grad_cam_img is in BGR format and uint8
                if grad_cam_img.ndim == 2:
                    grad_cam_img = cv2.cvtColor(grad_cam_img, cv2.COLOR_GRAY2BGR)
                elif grad_cam_img.shape[2] == 1:
                    grad_cam_img = cv2.cvtColor(grad_cam_img, cv2.COLOR_GRAY2BGR)

                # Convert all images to the same type and range
                original_img = original_img.astype(np.float32) / 255.0
                filtered_img = filtered_img.astype(np.float32) / 255.0
                grad_cam_img = grad_cam_img.astype(np.float32) / 255.0

                # 분류 결과를 텍스트로 추가
                label = 'OK' if target_class == 1 else 'NG'
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(original_img, f'Predicted: {label}', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # 이미지를 가로로 붙이기
                combined_img = np.hstack((original_img, filtered_img, grad_cam_img))
                combined_img = (combined_img * 255).astype(np.uint8)

                # 이미지 저장
                finalized_path = os.path.join(finalized_category_dir, filename)
                cv2.imwrite(finalized_path, combined_img)
                print(f"Saved combined image to {finalized_path}")
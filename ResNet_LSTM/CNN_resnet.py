import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# 폰트 관련 용도
import matplotlib.font_manager as fm

# 파이토치 관련 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# torchvision.models 임포트 (ResNet 사용을 위해)
import torchvision.models as models

# 디바이스 할당
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###########################################
# 1. 기본 CNN 모델 정의
###########################################

# 간단한 CNN의 레이어 정의
conv1 = nn.Conv2d(3, 32, 3)
relu = nn.ReLU(inplace=True)
conv2 = nn.Conv2d(32, 32, 3)
maxpool = nn.MaxPool2d((2,2))

# conv1의 weight[0]는 0번째 출력 채널의 가중치
w = conv1.weight[0]

# CNN의 feature extractor 정의
features = nn.Sequential(
    conv1,
    relu,
    conv2,
    relu,
    maxpool
)

# flatten 함수 정의
flatten = nn.Flatten()


###########################################
# 2. 학습 및 평가 관련 함수 정의
###########################################

# 손실 계산 함수
def eval_loss(loader, device, net, criterion):
    # 데이터로더에서 처음 한 개 세트를 가져옴
    for images, labels in loader:
        break

    # 디바이스 할당
    inputs = images.to(device)
    labels = labels.to(device)

    # 예측 계산
    outputs = net(inputs)

    # 손실 계산
    loss = criterion(outputs, labels)
    return loss

# 학습 함수
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
    from tqdm import tqdm  # 일반 tqdm 사용

    base_epochs = len(history)
    for epoch in range(base_epochs, num_epochs + base_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # 훈련 페이즈
        net.train()
        count = 0
        for inputs, labels in tqdm(train_loader, dynamic_ncols=True, leave=False):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 경사 초기화
            optimizer.zero_grad()

            # 예측 계산
            outputs = net(inputs)

            # 손실 계산
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 경사 계산 및 파라미터 업데이트
            loss.backward()
            optimizer.step()

            # 예측 라벨 산출 및 정답 건수 산출
            predicted = torch.max(outputs, 1)[1]
            train_acc += (predicted == labels).sum().item()

            # (각 배치마다 평균값은 마지막 배치 이후에 산출)

        avg_train_loss = train_loss / count
        avg_train_acc = train_acc / count

        # 검증 페이즈
        net.eval()
        count = 0
        for inputs, labels in test_loader:
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = torch.max(outputs, 1)[1]
            val_acc += (predicted == labels).sum().item()

        avg_val_loss = val_loss / count
        avg_val_acc = val_acc / count

        print(f'Epoch [{epoch+1}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history

# 학습 로그 해석 및 시각화 함수
def evaluate_history(history):
    print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}') 
    print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}')

    num_epochs = len(history)
    unit = num_epochs / 10

    # 손실 곡선
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='훈련')
    plt.plot(history[:,0], history[:,3], 'k', label='검증')
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('손실')
    plt.title('학습 곡선(손실)')
    plt.legend()
    plt.show()

    # 정확도 곡선
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='훈련')
    plt.plot(history[:,0], history[:,4], 'k', label='검증')
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('정확도')
    plt.title('학습 곡선(정확도)')
    plt.legend()
    plt.show()

# 이미지와 라벨 표시 함수
def show_images_labels(loader, classes, net, device):
    # 데이터로더에서 첫 세트를 가져옴
    for images, labels in loader:
        break
    n_size = min(len(images), 50)

    if net is not None:
        inputs = images.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        predicted = torch.max(outputs, 1)[1]

    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        if net is not None:
            predicted_name = classes[predicted[i]]
            c = 'k' if label_name == predicted_name else 'b'
            ax.set_title(f'{label_name}:{predicted_name}', c=c, fontsize=20)
        else:
            ax.set_title(label_name, fontsize=20)
        image_np = images[i].numpy().copy()
        img = np.transpose(image_np, (1, 2, 0))
        img = (img + 1) / 2
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()

# 파이토치 난수 고정 함수
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


###########################################
# 3. 데이터셋 및 데이터로더 준비 (CIFAR-10)
###########################################

# transformer1: 1계 텐서화 (평면 벡터 형태)
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.Lambda(lambda x: x.view(-1)),
])

# transformer2: 정규화만 실시 (이미지 형태 유지)
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

data_root = './data'

# CIFAR-10 훈련/검증 데이터셋 (두 가지 transformer 사용)
train_set1 = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform1)
test_set1  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform1)
train_set2 = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform2)
test_set2  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform2)

# 첫번째 샘플 확인
image1, label1 = train_set1[0]
image2, label2 = train_set2[0]

# 미니 배치 사이즈 지정
batch_size = 100

# 데이터로더 정의
train_loader1 = DataLoader(train_set1, batch_size=batch_size, shuffle=True)
test_loader1  = DataLoader(test_set1,  batch_size=batch_size, shuffle=False)
train_loader2 = DataLoader(train_set2, batch_size=batch_size, shuffle=True)
test_loader2  = DataLoader(test_set2,  batch_size=batch_size, shuffle=False)

# train_loader에서 한 세트 미리 가져오기 (디버깅용)
for images1, labels1 in train_loader1:
    break
for images2, labels2 in train_loader2:
    break

# 정답 라벨 정의
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 입력 차원수 (transform1 적용 시)
n_input = image1.view(-1).shape[0]

# 출력 차원수 (분류 클래스 수: 10)
n_output = len(set(list(labels1.data.numpy())))

# 은닉층의 노드 수 (CNN 모델 전용)
n_hidden = 128


###########################################
# 4. 모델 정의
###########################################

# 기존 CNN 모델
class CNN(nn.Module):
    def __init__(self, n_output, n_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(6272, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool
        )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3  

# ResNet 모델 (ResNet18 기반, CIFAR-10에 맞게 수정)
class ResNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetModel, self).__init__()
        # pretrained=False로 설정 (원하는 경우 True로 설정 후 fine-tuning 가능)
        self.model = models.resnet18(pretrained=False)
        # CIFAR-10 이미지 크기(32x32)에 맞게 첫번째 conv layer 수정
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 원래 maxpool layer를 Identity()로 대체하여 해상도 손실 최소화
        self.model.maxpool = nn.Identity()
        # fc layer를 CIFAR-10 클래스 수에 맞게 수정
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# 모델 선택: use_resnet가 True이면 ResNet, False이면 기존 CNN 사용
use_resnet = True

if use_resnet:
    net = ResNetModel(num_classes=n_output).to(device)
else:
    net = CNN(n_output, n_hidden).to(device)

# 손실 함수: 교차 엔트로피
criterion = nn.CrossEntropyLoss()
# 학습률 설정
lr = 0.01
# 최적화 함수: SGD
optimizer = optim.SGD(net.parameters(), lr=lr)


###########################################
# 5. 모델 학습 및 평가
###########################################

# 초기 손실 계산 (테스트 데이터 사용)
loss = eval_loss(test_loader2, device, net, criterion)
print("초기 손실:", loss.item())

# 난수 초기화 (재현성을 위해)
torch_seed()

# (이미 net, criterion, optimizer는 위에서 생성되었으므로 다시 생성하지 않습니다.)

# 반복 횟수 설정
num_epochs = 50
# 평가 결과 기록을 위한 배열 (열: epoch, train_loss, train_acc, val_loss, val_acc)
history2 = np.zeros((0, 5))

# 학습 진행
history2 = fit(net, optimizer, criterion, num_epochs, train_loader2, test_loader2, device, history2)

# 학습 로그 평가 및 시각화
evaluate_history(history2)

# 처음 50개 데이터 및 예측 결과 시각화
show_images_labels(test_loader2, classes, net, device)

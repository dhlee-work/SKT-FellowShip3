import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is av ailable")

learning_rate = 0.001
batch_size = 100
num_classes = 10
epochs = 5

# MNIST 데이터셋 로드
train_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transfroms.Compose([
        transfroms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)
test_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=False,
    download=True,
    transform=transfroms.Compose([
        transfroms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

# train_loader, test_loader 생성
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

# input size를 알기 위해서
examples = enumerate(train_set)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape


class ConvNet(nn.Module):
    def __init__(self):  # layer 정의
        super(ConvNet, self).__init__()

        # input size = 28x28
        self.conv1 = nn.Conv2d(1, 10,
                               kernel_size=5)  # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((W-K+2P)/S)+1 공식으로 인해 ((28-5+0)/1)+1=24 -> 24x24로 변환
        # maxpooling하면 12x12

        self.conv2 = nn.Conv2d(10, 20,
                               kernel_size=5)  # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((12-5+0)/1)+1=8 -> 8x8로 변환
        # maxpooling하면 4x4

        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)  # 랜덤하게 뉴런을 종료해서 학습을 방해해 학습이 학습용 데이터에 치우치는 현상을 막기 위해 사용
        self.mp = nn.MaxPool2d(2)  # 오버피팅을 방지하고, 연산에 들어가는 자원을 줄이기 위해 maxpolling
        self.fc1 = nn.Linear(320, 100)  # 4x4x20 vector로 flat한 것을 100개의 출력으로 변경
        self.fc2 = nn.Linear(100, 10)  # 100개의 출력을 10개의 출력으로 변경

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))  # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10
        x = F.relu(self.mp(self.conv2(x)))  # convolution layer 2번에 relu를 씌우고 maxpool, 결과값은 4x4x20
        x = self.drop2D(x)
        x = x.view(x.size(0), -1)  # flat
        x = self.fc1(x)  # fc1 레이어에 삽입
        x = self.fc2(x)  # fc2 레이어에 삽입
        return F.log_softmax(x)  # fully-connected layer에 넣고 logsoftmax 적용


model = ConvNet().to(device)  # CNN instance 생성
# Cost Function과 Optimizer 선택
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):  # epochs수만큼 반복
    avg_cost = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()  # 모든 model의 gradient 값을 0으로 설정
        hypothesis = model(data)  # 모델을 forward pass해 결과값 저장
        cost = criterion(hypothesis, target)  # output과 target의 loss 계산
        cost.backward()  # backward 함수를 호출해 gradient 계산
        optimizer.step()  # 모델의 학습 파라미터 갱신
        avg_cost += cost / len(train_loader)  # loss 값을 변수에 누적하고 train_loader의 개수로 나눔 = 평균
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# test
model.eval()  # evaluate mode로 전환 dropout 이나 batch_normalization 해제
with torch.no_grad():  # grad 해제
    correct = 0
    total = 0

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        preds = torch.max(out.data, 1)[1]  # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출
        total += len(target)  # 전체 클래스 개수
        correct += (preds == target).sum().item()  # 예측값과 실제값이 같은지 비교

    print('Test Accuracy: ', 100. * correct / total, '%')

# moduDeep_pytorch with Docker🐳

- PyTorch 1.10.1
  - ![image](https://user-images.githubusercontent.com/72531381/148143832-2612c922-c05d-4dca-810b-c5e265d5820b.jpeg)
  
  - visual studio code 에서 사용하려고,
  ```
   Python 2.7.17이 깔려있고 torch가 3.7 이상 3.9 이하 사이의 버전과 호환되기 때문에
   ctrl + shift + p 로 select interpreter 를 열어
   Python 3.9.7 64-bit ('base': conda) 를 선택하였다. 
  ```
- torchvision 0.11.2  
- conda 4.11.0 

- cudatoolkit 10.2
![image](https://user-images.githubusercontent.com/72531381/148143486-e737860d-d6b3-4fe7-bc62-5c042796ef59.jpeg)

- python3 -m venv ./name

---
yolov3
v1,v2 는 review
resnet 까지 완료!!!!
batch normalization 읽기!
fast R cnn 도전!
mobileNet ShuffleNet : 초경량 모델

 - 핸드폰이나 임베디드 시스템 같이 저용량 메모리환경에 딥러닝을 적용하기 위해서는 모델 경량화가 필요하다.
 - 메모리가 제한된 환경에서 MobileNet 을 최적으로 

 -  ShuffleNet은 경량화에 집중한 모델. 또 다른 경량화 모델인 MobileNetV1을 능가. 
 - 그리고 AlexNet과 비교하여 동일한 정확도로 13배나 더 빠르다. 
 NasNet & EfficentNet
경량화에 대한 다양한 기법들
object detection(+transformer 기법까지)
---

## 목차(Part 1 ~ Part 3 / Part 4는 하지 않음)
  - PART 1: Machine Learning & PyTorch Basic
    - Lab-01-1 Tensor Manipulation 1
    - Lab-01-2 Tensor Manipulation 2
    - Lab-02 Linear regression
    - Lab-03 Deeper Look at GD
    - Lab-04-1 Multivariable Linear regression
    - Lab-04-2 Loading Data
    - Lab-05 Logistic Regression
    - Lab-06 Softmax Classification
    - Lab-07-1 Tips
    - Lab-07-2 MNIST Introduction
  - PART 2: Neural Network
    - Lab-08-1 Perceptron
    - Lab-08-2 Multi Layer Perceptron
    - Lab-09-1 ReLU
    - Lab-09-2 Weight initialization
    - Lab-09-3 Dropout
    - Lab-09-4 Batch Normalization
  - PART 3: Convolutional Neural Network
    - Lab-10-0 Convolution Neural Networkintro
    - Lab-10-1 Convolution
    - Lab-10-2 mnist cnn
    - Lab-10-3 visdom
    - Lab-10-4-1 ImageFolder1
    - Lab-10-4-2 ImageFolder2
    - Lab-10-5 Advance CNN(VGG)
    - Lab-10-6-1 Advanced CNN(RESNET-1)
    - Lab-10-6-2 Advanced CNN(RESNET-2)
    - Lab-10-7 Next step of CNN
  - PART 4: Recurrent Neural Network
    - Lab-11-0 RNN intro
    - Lab-11-1 RNN basics
    - Lab-11-2 RNN hihello and charseq
    - Lab-11-3 Long sequence
    - Lab-11-4 RNN timeseries
    - Lab-11-5 RNN seq2seq
    - Lab-11-6 PackedSequence
---

## Dokcer 🐳 란,

- 컨테이너 기반의 가상화 시스템.
- 다양한 프로그램, 실행환경을 컨테이너로 추상화하고 동일한 인터페이스를 제공하여 
프로그램의 배포 및 관리를 단순하게 해줍니다. 백엔드 프로그램, 데이터베이스 서버, 메시지 큐등 
어떤 프로그램도 컨테이너로 추상화할 수 있고 조립PC, AWS, Azure, Google cloud등 어디에서든 실행할 수 있다.

## Lab-01-1 Tensor Manipulation 1️⃣

- Vector, Matrix and Tensor
  - 1차원 : Vector
  - 2차원 : Matrix
  - 3차원 : Tensor
  - 4,5,6차원
  ![image](https://user-images.githubusercontent.com/72531381/147435475-b85f293e-204e-4f94-a0cf-400918778a8d.jpg)
- Numpy Review
- PyTorch Tensor Allocation
- Matrix Multiplication
- Other Basic Ops


---
---
## Ubuntu 18.04.5 LTS


- 토 나오는 설치과정..

- 무선,유선 연결되지 않은 상태에서 우분투를 설치하니.. 랜드라이버와 인터넷 드라이버가 잡히지 않았다. 사무실 본체 랜카드는 (인텔 I219-V) 이고, 인텔 홈페이지에서 리눅스용 드라이버를 다운받았다. 수동으로 설치하려고 하니 make 명령어를 통해 설치를 진행해야 한다.

- 그러나 make 은 기본 패키지가 아니기 때문에 패키지를 수동으로 설치하는 방법을 사용하겠다.

- 먼저 인터넷이 가능한 곳에서 같은 버전의 우분투 18.04를 설치하고 필요한 패키지를 다운받았다. 다운받은 패키지들을 압축하여, 오프라인인 우분투 환경에 /var/cache/apt/archives 경로로 풀었다. dpkg --force-all -i *.deb 으로 강제로 설치하였으나 결과는 아래 그림과 같이 실패하였다. 

![image](https://user-images.githubusercontent.com/72531381/147816097-d1e9abe6-2ce5-497a-ab77-9e56dae5feca.jpg)

- 꼭 해결하고 만다.

- 와!!!!!!!!!!!! 
- 해결방법은 usb 랜포트를 사용하니 갑자기 랜을 잡았다!!
- 이제 cuda 10.02 cudnn 7.6.5
- lan 카드 32gb
- intel corporation device 15fa 용 리눅스 드라이버 설치를 완료하였다!!

---
## Memo
- ### torch.nn
- PyTorch는 랜덤 또는 0으로만 이루어진 텐서를 생성하는 메서드를 제공하고, 
- 우리는 간단한 선형 모델의 가중치(weights)와 절편(bias)을 생성하기 위해서 이것을 사용할 것이다.
- 이들은 일반적인 텐서에 매우 특별한 한 가지가 추가된 것이다.
- PyTorch에게 이들이 기울기(gradient)가 필요하다고 알려준다.
- 이를 통해 PyTorch는 텐서에 행해지는 모든 연산을 기록하게 하고, 따라서 자동적으로 역전파(back-propagation) 동안에 기울기를 계산할 수 있다.

- 활성화 함수가 대체 어떻게 쓰이는가? 
- 크로스 엔트로피!!
- 러닝 레이트 조절 하는 것은 어떻게 하는가?
- 테스트 데이터 유효데이터 나누는 이유?
- 배치 사이즈!! 배치 정규화는 어케 되는가!!!

- ### numpy
  - numpy의 class ndarray
  - ndarray.ndim : 배열의 차원 수 or 배열의 축 수
  - ndarray.shape : 배열 각 차원의 크기를 튜플 형태로 표현.
  - ndarray.dtype : 원소의 종류, 기본적으로 넘파이의 원소들은 모두 같은 데이터 타입
  - Ellipsis Slicing (Array Slicing - Three dot):
    - [ : ], [ 1 , : ] 과 같이 : 콜론 표현은 '해당 차원의 모든 원소'를 의미한다.
    - Ellipsis Slicing 또한 : 콜론과 같은 것을 의미하는데 다차원에 대해 여러개의 : 콜론을 사용하는 것을 ... 하나로 생략할 수 있다.
    - 예를들어 a[:,2:]가 모든 열에 대해 2번째 행부터 마지막 행까지 Slice한다면 이것은 a[...,2:] 라고 쓸 수 있다.
    ```python
    >>> a[...]
    array([[1, 2, 3],
          [2, 3, 4],
          [3, 4, 5]])
    >>> a[1:, ...]
    array([[2, 3, 4],
          [3, 4, 5]])
    >>> a[1:, ...]
    array([2, 3])
    ```

- ### Tensor
  - 데이터 생성
  ```python
  # 직접 생성
  data = [[1,2],[3,4]] # List
  x_data = torch.tensor(data) # Matrix == 2D tensor

  # numpy 배열로부터 생성
  np_arr = np.array(data) # 2차 배열
  x_np = torch.from_numpy(np_array) # Matrix == 2D tensor

  # 다른 텐서로부터 생성
  x_ones = torch.ones_like(x_data)
  print(f"Ones Tensor: \n {x_ones} \n")

  # 무작위(random) 또는 상수(Constant) 값을 사용하기
  shape = (2,3,)
  print(len(shape))
  print(type(shape))
  rand_tensor = torch.rand(shape)
  ones_tensor = torch.ones(shape)
  zeros_tensor = torch.zeros(shape)

  print(f"Random Tensor: \n {rand_tensor} \n")
  print(f"Ones Tensor: \n {ones_tensor} \n")
  print(f"Zeros Tensor: \n {zeros_tensor}")
  ```
<<<<<<< HEAD

- ### CNN
  1. Convolution(합성곱)
  2. 채널(Channel)
  3. 필터(Filter)
  4. 커널(Kernel)
  5. 스트라이드(Strid)
  6. 패딩(Padding)
  7. 피처 맵(Feature Map)
  8. 액티베이션 맵(Activation Map)
  9. 풀링(Pooling) 레이어
=======
>>>>>>> 26ff45c4d690e5ec45baf2bad06578f07015e033

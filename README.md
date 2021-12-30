# moduDeep_pytorch with Docker🐳

- PyTorch 1.10.1
  - ![image](https://user-images.githubusercontent.com/72531381/147450783-5c5cd01d-328e-4fec-ba4d-a0cb2b10963b.jpg)
  
  - visual studio code 에서 사용하려고,
  ```
   ctrl + shift + p 해서 select interpreter 를 열어
   Python 3.9.7 64-bit ('base': conda) 를 선택하였다.
  ```
- conda 4.11.0

- cudatoolkit 11.3.1
![image](https://user-images.githubusercontent.com/72531381/147450745-123915bd-4712-4d6b-b22f-cd89fd2ac434.jpg)


---
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
# Ubuntu 18.04.06 LTS

- 토 나오는 설치과정..
- 무선,유선 연결되지 않은 상태에서 우분투를 설치하니.. 랜드라이버와 인터넷 드라이버가 잡히지 않으며.. 이것을 수동으로 설치하려고 하니 make 명령어도 깔려있지 않아 죽을 맛이다.
- apt-get install gcc make 을 수동으로 설치할 수 있어야 한다.

- 이더넷 드라이버를 설치..I219-V

- virtualbox 에서 패키지를 옮기려 했으나..
- 명령어가 먹지 않았다.
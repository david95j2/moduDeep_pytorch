# moduDeep_pytorch with Docker๐ณ

- PyTorch 1.10.1
  - ![image](https://user-images.githubusercontent.com/72531381/148143832-2612c922-c05d-4dca-810b-c5e265d5820b.jpeg)
  
  - visual studio code ์์ ์ฌ์ฉํ๋ ค๊ณ ,
  ```
   Python 2.7.17์ด ๊น๋ ค์๊ณ  torch๊ฐ 3.7 ์ด์ 3.9 ์ดํ ์ฌ์ด์ ๋ฒ์ ๊ณผ ํธํ๋๊ธฐ ๋๋ฌธ์
   ctrl + shift + p ๋ก select interpreter ๋ฅผ ์ด์ด
   Python 3.9.7 64-bit ('base': conda) ๋ฅผ ์ ํํ์๋ค. 
  ```
- torchvision 0.11.2  
- conda 4.11.0 

- cudatoolkit 10.2
![image](https://user-images.githubusercontent.com/72531381/148143486-e737860d-d6b3-4fe7-bc62-5c042796ef59.jpeg)

- python3 -m venv ./name

---
yolov3 ์ผ๋จ์ ์๋ review + open source
v1,v2 ๋ review
resnet ๊น์ง ์๋ฃ!!!!
batch normalization ์ฝ๊ธฐ!
fast R cnn ๋์ !
mobileNet ShuffleNet : ์ด๊ฒฝ๋ ๋ชจ๋ธ

 - ํธ๋ํฐ์ด๋ ์๋ฒ ๋๋ ์์คํ ๊ฐ์ด ์ ์ฉ๋ ๋ฉ๋ชจ๋ฆฌํ๊ฒฝ์ ๋ฅ๋ฌ๋์ ์ ์ฉํ๊ธฐ ์ํด์๋ ๋ชจ๋ธ ๊ฒฝ๋ํ๊ฐ ํ์ํ๋ค.
 - ๋ฉ๋ชจ๋ฆฌ๊ฐ ์ ํ๋ ํ๊ฒฝ์์ MobileNet ์ ์ต์ ์ผ๋ก 

 -  ShuffleNet์ ๊ฒฝ๋ํ์ ์ง์คํ ๋ชจ๋ธ. ๋ ๋ค๋ฅธ ๊ฒฝ๋ํ ๋ชจ๋ธ์ธ MobileNetV1์ ๋ฅ๊ฐ. 
 - ๊ทธ๋ฆฌ๊ณ  AlexNet๊ณผ ๋น๊ตํ์ฌ ๋์ผํ ์ ํ๋๋ก 13๋ฐฐ๋ ๋ ๋น ๋ฅด๋ค. 
 NasNet & EfficentNet
๊ฒฝ๋ํ์ ๋ํ ๋ค์ํ ๊ธฐ๋ฒ๋ค
object detection(+transformer ๊ธฐ๋ฒ๊น์ง)
---

## ๋ชฉ์ฐจ(Part 1 ~ Part 3 / Part 4๋ ํ์ง ์์)
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

## Dokcer ๐ณ ๋,

- ์ปจํ์ด๋ ๊ธฐ๋ฐ์ ๊ฐ์ํ ์์คํ.
- ๋ค์ํ ํ๋ก๊ทธ๋จ, ์คํํ๊ฒฝ์ ์ปจํ์ด๋๋ก ์ถ์ํํ๊ณ  ๋์ผํ ์ธํฐํ์ด์ค๋ฅผ ์ ๊ณตํ์ฌ 
ํ๋ก๊ทธ๋จ์ ๋ฐฐํฌ ๋ฐ ๊ด๋ฆฌ๋ฅผ ๋จ์ํ๊ฒ ํด์ค๋๋ค. ๋ฐฑ์๋ ํ๋ก๊ทธ๋จ, ๋ฐ์ดํฐ๋ฒ ์ด์ค ์๋ฒ, ๋ฉ์์ง ํ๋ฑ 
์ด๋ค ํ๋ก๊ทธ๋จ๋ ์ปจํ์ด๋๋ก ์ถ์ํํ  ์ ์๊ณ  ์กฐ๋ฆฝPC, AWS, Azure, Google cloud๋ฑ ์ด๋์์๋  ์คํํ  ์ ์๋ค.

## Lab-01-1 Tensor Manipulation 1๏ธโฃ

- Vector, Matrix and Tensor
  - 1์ฐจ์ : Vector
  - 2์ฐจ์ : Matrix
  - 3์ฐจ์ : Tensor
  - 4,5,6์ฐจ์
  ![image](https://user-images.githubusercontent.com/72531381/147435475-b85f293e-204e-4f94-a0cf-400918778a8d.jpg)
- Numpy Review
- PyTorch Tensor Allocation
- Matrix Multiplication
- Other Basic Ops


---
---
## Ubuntu 18.04.5 LTS


- ํ  ๋์ค๋ ์ค์น๊ณผ์ ..

- ๋ฌด์ ,์ ์  ์ฐ๊ฒฐ๋์ง ์์ ์ํ์์ ์ฐ๋ถํฌ๋ฅผ ์ค์นํ๋.. ๋๋๋ผ์ด๋ฒ์ ์ธํฐ๋ท ๋๋ผ์ด๋ฒ๊ฐ ์กํ์ง ์์๋ค. ์ฌ๋ฌด์ค ๋ณธ์ฒด ๋์นด๋๋ (์ธํ I219-V) ์ด๊ณ , ์ธํ ํํ์ด์ง์์ ๋ฆฌ๋์ค์ฉ ๋๋ผ์ด๋ฒ๋ฅผ ๋ค์ด๋ฐ์๋ค. ์๋์ผ๋ก ์ค์นํ๋ ค๊ณ  ํ๋ make ๋ช๋ น์ด๋ฅผ ํตํด ์ค์น๋ฅผ ์งํํด์ผ ํ๋ค.

- ๊ทธ๋ฌ๋ make ์ ๊ธฐ๋ณธ ํจํค์ง๊ฐ ์๋๊ธฐ ๋๋ฌธ์ ํจํค์ง๋ฅผ ์๋์ผ๋ก ์ค์นํ๋ ๋ฐฉ๋ฒ์ ์ฌ์ฉํ๊ฒ ๋ค.

- ๋จผ์  ์ธํฐ๋ท์ด ๊ฐ๋ฅํ ๊ณณ์์ ๊ฐ์ ๋ฒ์ ์ ์ฐ๋ถํฌ 18.04๋ฅผ ์ค์นํ๊ณ  ํ์ํ ํจํค์ง๋ฅผ ๋ค์ด๋ฐ์๋ค. ๋ค์ด๋ฐ์ ํจํค์ง๋ค์ ์์ถํ์ฌ, ์คํ๋ผ์ธ์ธ ์ฐ๋ถํฌ ํ๊ฒฝ์ /var/cache/apt/archives ๊ฒฝ๋ก๋ก ํ์๋ค. dpkg --force-all -i *.deb ์ผ๋ก ๊ฐ์ ๋ก ์ค์นํ์์ผ๋ ๊ฒฐ๊ณผ๋ ์๋ ๊ทธ๋ฆผ๊ณผ ๊ฐ์ด ์คํจํ์๋ค. 

![image](https://user-images.githubusercontent.com/72531381/147816097-d1e9abe6-2ce5-497a-ab77-9e56dae5feca.jpg)

- ๊ผญ ํด๊ฒฐํ๊ณ  ๋ง๋ค.

- ์!!!!!!!!!!!! 
- ํด๊ฒฐ๋ฐฉ๋ฒ์ usb ๋ํฌํธ๋ฅผ ์ฌ์ฉํ๋ ๊ฐ์๊ธฐ ๋์ ์ก์๋ค!!
- ์ด์  cuda 10.02 cudnn 7.6.5
- lan ์นด๋ 32gb
- intel corporation device 15fa ์ฉ ๋ฆฌ๋์ค ๋๋ผ์ด๋ฒ ์ค์น๋ฅผ ์๋ฃํ์๋ค!!

---
## Memo
- ### torch.nn
- PyTorch๋ ๋๋ค ๋๋ 0์ผ๋ก๋ง ์ด๋ฃจ์ด์ง ํ์๋ฅผ ์์ฑํ๋ ๋ฉ์๋๋ฅผ ์ ๊ณตํ๊ณ , 
- ์ฐ๋ฆฌ๋ ๊ฐ๋จํ ์ ํ ๋ชจ๋ธ์ ๊ฐ์ค์น(weights)์ ์ ํธ(bias)์ ์์ฑํ๊ธฐ ์ํด์ ์ด๊ฒ์ ์ฌ์ฉํ  ๊ฒ์ด๋ค.
- ์ด๋ค์ ์ผ๋ฐ์ ์ธ ํ์์ ๋งค์ฐ ํน๋ณํ ํ ๊ฐ์ง๊ฐ ์ถ๊ฐ๋ ๊ฒ์ด๋ค.
- PyTorch์๊ฒ ์ด๋ค์ด ๊ธฐ์ธ๊ธฐ(gradient)๊ฐ ํ์ํ๋ค๊ณ  ์๋ ค์ค๋ค.
- ์ด๋ฅผ ํตํด PyTorch๋ ํ์์ ํํด์ง๋ ๋ชจ๋  ์ฐ์ฐ์ ๊ธฐ๋กํ๊ฒ ํ๊ณ , ๋ฐ๋ผ์ ์๋์ ์ผ๋ก ์ญ์ ํ(back-propagation) ๋์์ ๊ธฐ์ธ๊ธฐ๋ฅผ ๊ณ์ฐํ  ์ ์๋ค.

- ํ์ฑํ ํจ์๊ฐ ๋์ฒด ์ด๋ป๊ฒ ์ฐ์ด๋๊ฐ? 
- ํฌ๋ก์ค ์ํธ๋กํผ!!
- ๋ฌ๋ ๋ ์ดํธ ์กฐ์  ํ๋ ๊ฒ์ ์ด๋ป๊ฒ ํ๋๊ฐ?
- ํ์คํธ ๋ฐ์ดํฐ ์ ํจ๋ฐ์ดํฐ ๋๋๋ ์ด์ ?
- ๋ฐฐ์น ์ฌ์ด์ฆ!! ๋ฐฐ์น ์ ๊ทํ๋ ์ด์ผ ๋๋๊ฐ!!!

- ### numpy
  - numpy์ class ndarray
  - ndarray.ndim : ๋ฐฐ์ด์ ์ฐจ์ ์ or ๋ฐฐ์ด์ ์ถ ์
  - ndarray.shape : ๋ฐฐ์ด ๊ฐ ์ฐจ์์ ํฌ๊ธฐ๋ฅผ ํํ ํํ๋ก ํํ.
  - ndarray.dtype : ์์์ ์ข๋ฅ, ๊ธฐ๋ณธ์ ์ผ๋ก ๋ํ์ด์ ์์๋ค์ ๋ชจ๋ ๊ฐ์ ๋ฐ์ดํฐ ํ์
  - Ellipsis Slicing (Array Slicing - Three dot):
    - [ : ], [ 1 , : ] ๊ณผ ๊ฐ์ด : ์ฝ๋ก  ํํ์ 'ํด๋น ์ฐจ์์ ๋ชจ๋  ์์'๋ฅผ ์๋ฏธํ๋ค.
    - Ellipsis Slicing ๋ํ : ์ฝ๋ก ๊ณผ ๊ฐ์ ๊ฒ์ ์๋ฏธํ๋๋ฐ ๋ค์ฐจ์์ ๋ํด ์ฌ๋ฌ๊ฐ์ : ์ฝ๋ก ์ ์ฌ์ฉํ๋ ๊ฒ์ ... ํ๋๋ก ์๋ตํ  ์ ์๋ค.
    - ์๋ฅผ๋ค์ด a[:,2:]๊ฐ ๋ชจ๋  ์ด์ ๋ํด 2๋ฒ์งธ ํ๋ถํฐ ๋ง์ง๋ง ํ๊น์ง Sliceํ๋ค๋ฉด ์ด๊ฒ์ a[...,2:] ๋ผ๊ณ  ์ธ ์ ์๋ค.
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
  - ๋ฐ์ดํฐ ์์ฑ
  ```python
  # ์ง์  ์์ฑ
  data = [[1,2],[3,4]] # List
  x_data = torch.tensor(data) # Matrix == 2D tensor

  # numpy ๋ฐฐ์ด๋ก๋ถํฐ ์์ฑ
  np_arr = np.array(data) # 2์ฐจ ๋ฐฐ์ด
  x_np = torch.from_numpy(np_array) # Matrix == 2D tensor

  # ๋ค๋ฅธ ํ์๋ก๋ถํฐ ์์ฑ
  x_ones = torch.ones_like(x_data)
  print(f"Ones Tensor: \n {x_ones} \n")

  # ๋ฌด์์(random) ๋๋ ์์(Constant) ๊ฐ์ ์ฌ์ฉํ๊ธฐ
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
  1. Convolution(ํฉ์ฑ๊ณฑ)
  2. ์ฑ๋(Channel)
  3. ํํฐ(Filter)
  4. ์ปค๋(Kernel)
  5. ์คํธ๋ผ์ด๋(Strid)
  6. ํจ๋ฉ(Padding)
  7. ํผ์ฒ ๋งต(Feature Map)
  8. ์กํฐ๋ฒ ์ด์ ๋งต(Activation Map)
  9. ํ๋ง(Pooling) ๋ ์ด์ด
=======
>>>>>>> 26ff45c4d690e5ec45baf2bad06578f07015e033

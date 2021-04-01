# AI_PlayGround

## 1. Understanding Generative Adversarial Networks (GANs)
### **0) GANs의 기본 아이디어**</br>
**GANs**는 진품 와인만을 매입하려는 '와인 판매업자'와, 가짜 와인을 주조하여 와인 판매업자를 속이려는 '와인 위조범'과의 싸움으로 비유할 수 있다.</br></br>
![image](https://user-images.githubusercontent.com/76294398/112935056-f565af00-915d-11eb-8718-31aab5105cb5.png)</br>
위조범은 처음에는 단지 색만 비슷한 가짜 와인을 주조했을 것이고, 이는 금방 탄로났을 것이다. 하지만 이 위조범은 여기서 멈추지 않는다. 자신이 주조한 가짜 와인이 왜 와인 판매업자를 속이지 못했는지 이유를 분석하여 결점을 계속해서 보완해 나갈 것이다. 이에 맞춰 와인 판매업자 또한 와인의 품질을 감별하는 절차를 계속해서 발전시킬 것이다.</br>
**이처럼, 거짓된 정보를 진실로 믿게하려는 '공격자'와 거짓을 거짓으로 판별하려는 '방어자'의 싸움이 바로 GANs의 기본 아이디어이다.**</br></br>

### **1) GANs의 구성요소**</br>

GANs의 주요 구성요소로는 **Generator(생성자)** 와 **Discriminator(식별자)** 가 있다.</br>
위 비유에서, 가짜 와인을 주조(생성)하는 '와인 위조범'은 Generator, 그리고 이를 식별하는 '와인 판매업자'는 Discriminator라고 할 수 있다.</br></br>
![image](https://user-images.githubusercontent.com/76294398/112935573-fc40f180-915e-11eb-934d-51c2b1f1bded.png)</br>
이때 Discriminator가 해당 값의 진위여부를 판별할 수 있도록 확률값을 주는 네트워크로 **Convolutional Neural Networks** 를 사용한다.</br>
Generator 역시 확률 네트워크로 deconvolutional layer와 함께 Convolutional Neural Network를 이용하는데, 이 네트워크는 **Noise vector** 를 가져와서 이미지를 출력한다.</br>
Generative 네트워크를 훈련할 때, Discriminator가 생성된 이미지를 실제 이미지와 구분하는 데 어려움을 겪도록 이미지에서 개선/변경 영역을 학습한다.</br>

Generative 네트워크는 실제 이미지와 비슷한 모습을 계속해서 생성하는 반면, Discriminative 네트워크는 실제 이미지와 가짜 이미지의 차이를 확인하려고 노력한다. Generative 네트워크의 궁극적인 목표는 실제 이미지와 구별할 수 없는 이미지를 생성할 수 있는 생성 네트워크를 갖는 것이다.</br></br>

## **2) 케라스로 만드는 간단한 GANs**</br>

이제 GANs이 무엇이고 그 주요 구성 요소가 무엇인지 이해했으니, 케라스를 이용하여 아주 간단한 코드를 직접 만들어 보도록 하자.</br>

가장 먼저 해야될 일은 pip을 이용해서 다음의 패키지들을 다운 받는 것이다. 구글 colab에서 pip을 이용하기 위해서는 !pip을 이용하면 된다.</br>

- Keras
- matplotlib
- Tensorflow
- tqdm

```python
!pip install keras matplotlib tensorflow tqdm
```

<br/>
matplotlib는 plot을 그리는데 사용할 것이고, Tensorflow는 Keras의 Back-End로 이용할 것이다. 또한 tqdm을 이용해서 모델이 각 epoch 마다 어떠한 결과를 내고 있는지 보여줄 것이다.
<br/>
또한, 생성된 이미지를 저장하기위해 구글드라이브를 연동시켰다.
</br>

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from google.colab import drive
drive.mount('/content/gdrive')

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
```
<br/>
이제 변수를 설정하도록 하자.

```python
# Keras 가 Tensorflow 를 벡엔드로 사용할 수 있도록 설정
os.environ["KERAS_BACKEND"] = "tensorflow"

# seed 설정
np.random.seed(10)

# 랜덤 노이즈 벡터 차원 설정
random_dim = 100
```

Discriminator와 Generator를 만들기 전에 먼저 데이터를 수집하고 전처리해야 한다. <br/>
이번 실습에서는 0 부터 9 까지의 단일 자릿수의 이미지 세트인 데이터셋 MNIST를 사용할 것이다.
<br/>
![image](https://user-images.githubusercontent.com/76294398/112952203-33bb9800-9177-11eb-9781-e5d06f6fc395.png)

```python
def load_minst_data():
    # 데이터 로드
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 데이터를 -1 ~ 1 사이 값으로 normalize
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    # x_train을 (60000, 28, 28) 에서 (60000, 784) 로 reshape
    # 한 row 당 784 columns
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)
```
*Keras는 MNIST 데이터셋을 쉽게 이용할 수 있도록 mnist.load_data() 메서드를 기본으로 제공한다.
<br/><br/>

이제 Generator 및 Discriminatora 네트워크를 만들어 볼 것이다. 두 네트워크에 모두 Adam Optimizer를 사용한다. <br/>
Generator와 Discriminator의 경우 모두 세 개의 숨겨진 레이어가 있는 신경 네트워크를 생성하며 activation function(활성 함수)은 Leaky Relu를 사용한다. 또한 Discriminator가 보이지 않는 영상에서 견고성을 향상시킬 수 있도록 Dropout(드롭아웃) 레이어를 추가해야 한다.<br/>

```python
# Adam Optimizer를 사용
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

# Generator 만들기
def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

# Discriminator 만들기
def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator
```
<br/>
이제 Generator와 Discriminator를 함께 모아보도록 하자.<br/>

```python
def get_gan_network(discriminator, random_dim, generator, optimizer):
    # Generator와 Discriminator를 동시에 학습시키고 싶을 때 => trainable = False
    discriminator.trainable = False

    gan_input = Input(shape=(random_dim,))

    # Generator의 결과는 이미지
    x = generator(gan_input)

    # Discriminator의 결과는 이미지가 진짜인지 가짜인지에 대한 확률
    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan
```
<br/>

결과의 원활한 확인을 위해 20 epoch 마다 이미지를 생성하여 content/gan_images 폴더에 저장하도록 하자. <br/>

```python
# 생성된 MNIST 이미지를 보여주는 함수
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('/content/gan_images/gan_generated_image_epoch_%d.png' % epoch)
```
<br/>
이제 완성된 네트워크를 학습시키는 일만이 남았다. 학습이 완료된 후 결과를 확인하도록 하자.<br/>

```python
def train(epochs=1, batch_size=128):
    # train 데이터와 test 데이터 load
    x_train, y_train, x_test, y_test = load_minst_data()

    # train 데이터를 128 사이즈의 batch 로 분할
    batch_count = x_train.shape[0] // batch_size

    # GAN 네트워크 형성
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            # 입력으로 사용할 random 노이즈와 이미지 load
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])

            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # MNIST 이미지 생성
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9

            # Discriminator 학습
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Generator 학습
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            plot_generated_images(e, generator)

if __name__ == '__main__':
    train(400, 128)
```

![image](https://user-images.githubusercontent.com/76294398/112953274-50a49b00-9178-11eb-917e-7b0747398620.png)

<br/>
400 epochs 학습 후 생성된 이미지를 확인해보자. 1 epoch 이후에 생성된 이미지를 보면, 실질적인 구조가 없다는 것을 알 수 있다. 하지만 40 epochs 이후의 이미지를 보면, 숫자가 형성되기 시작하였고, 400 epochs 후에 생성된 이미지는 숫자가 뚜렷하게 나타남을 확인할 수 있다.<br/>

![image](https://user-images.githubusercontent.com/76294398/112953792-ba24a980-9178-11eb-96a4-65b052a44e65.png)

<br/><br/>

## **3) 결론**<br/>

이번 튜토리얼을 통해, 우리는 GAN의 기본을 수학적인 수식 없이 직관적으로 학습하였다. 또한 Keras 라이브러리의 도움을 받아 첫 번째 모델을 구현해냈다.<br/>
좀더 정확한 공부를 위해서는 수학적인 공식과 전반적인 모델을 이해할 필요가 있다.<br/>
새로운 레이어를 추가하여 더욱 복잡한 아키텍쳐를 형성해보거나, GPU 가속 등을 사용하는 방법도 공부할 수 있도록 해야겠다.

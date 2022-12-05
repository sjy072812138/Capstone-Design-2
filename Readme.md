# 캡스톤디자인2
컴퓨터공학과_2018102156_SUNJIYAO
## 얼굴인식 프로젝트

### 연구 배경

과학 기술의 발달로 얼굴 인식은 생활에서 없어서는 안될 것이 되었습니다.스마트폰, 출입문, 은행 관련 앱, 출석체크 등이 곳곳에서 얼굴인식을 볼 수 있다. 안면 인식에 대한 사람들의 필요점이 계속 증가하고 있다.얼굴인식 기술도 갈수록 중요해지고 있다.본 주제는 바로 딥러닝을 기반으로 얼굴 인식 기능을 개발하는 것입니다.
***

### 관련연구

#### python
간결한 문법으로 입문자가 이해하기 쉽고, 다양한 분야에 활용할 수 있기 때문이다. 이 외에도 파이썬은 머신러닝, 그래픽, 웹 개발 등 여러 업계에서 선호하는 언어로 꾸준히 성장하고 있다.

#### openCV
OpenCV(Open Source Computer Vision)은 실시간 컴퓨터 비전을 목적으로 한 프로그래밍 라이브러리이다. 원래는 인텔이 개발하였다. 실시간 이미지 프로세싱에 중점을 둔 라이브러리이다.OpenCV는 C/C++ 프로그래밍 언어로 개발 되었으며 파이썬, 자바 및 매트랩 / OCTAVE에 바인딩 되어 프로그래머에게 개발 환경을 지원합니다.

#### dlib
dlib는 머신러닝의 오픈 소스 라이브러리로 머신러닝의 많은 알고리즘을 포함하고 사용이 편리하며 헤더 파일을 직접 포함할 수 있으며 다른 라이브러리에 의존하지 않습니다.Dlib은 실제 문제를 해결하는 데 도움이 되는 복잡한 머신러닝 소프트웨어를 많이 만들 수 있도록 돕는다.현재 dlib는 로봇, 임베디드 기기, 휴대폰, 대규모 고성능 컴퓨팅 환경 등 산업·학술 분야에서 널리 활용되고 있다.

#### mysql
표준 데이터베이스 질의 언어인 구조화 질의 언어(SQL: Structured Query Language)를 사용하는 공개 소스의 관계형 데이터베이스 관리 시스템(RDBMS). 매우 빠르고, 유연하며, 사용하기 쉬운 특징이 있다. 다중 사용자, 다중 스레드(thread)를 지원하고, C, C++, 에펠(Eiffel), 자바, 펄, PHP, 파이선(Python) 스크립트 등을 위한 응용 프로그램 인터페이스(API)를 제공한다. 유닉스나 리눅스, 윈도우 운영 체제 등에서 사용할 수 있다. 램프(LAMP), 즉 리눅스 운영 체계와 아파치(Apache) 서버 프로그램, MySQL, PHP 스크립트 언어 구성은 상호 연동이 잘되면서도 공개 소스(오픈 소스)로 개발되는 무료 프로그램이어서 홈 페이지나 쇼핑몰 등 일반적인 웹 개발에 널리 이용되고 있다.

#### shufflnet
셔플넷(ShuffleNet)은 최근에 제안된 계산 효율적인 CNN 모델이다.ShuffleNet의 설계 목표는 제한된 계산 자원을 사용하여 최상의 모델 정확도를 달성하는 방법이며, 이는 속도와 정확도 간의 균형을 잘 맞추어야 합니다.ShuffleNet의 핵심은 pointwise group convolution과 channel shuffle의 두 가지 작업을 채택하여 정확도를 유지하면서 모델의 계산량을 크게 줄였습니다.

#### resnet
resnet은 최적화가 쉽고 상당한 깊이를 늘려 정확도를 높일 수 있는 것이 특징이다.그 내부의 잔차 블록은 점프를 이용하여 연결하였다.심층 신경망에서 깊이 증가에 따른 구배 소실 문제를 완화했다.
***
### 제안 연구의 중요성/독창성

사회의 발전에 따라서 인공지능은 우리 삶에 등장했다.특히 얼굴 인식은 다양한 분야에서 활용되고 있다.그래서 딥러닝을 통한 안면인식 관련 기술이 점점 더 중요해지고 있다.딥러닝을 이용하여 사용자가 본인인지 아닌지를 정확하게 판단할 수 있다.본 주제는 바로 딥러닝을 기반으로 얼굴 인식 기능을 개발하는 것입니다.본 프로젝트는 Tensorflow를 기반으로 두 개의 신경망(shuffllenet and resnet)을 결합하는 방법으로 얼굴인식 프로그램을 제작합니다.미래에는 캠퍼스, 자동차 등의 분야에 응용할 수 있다.이번에 제작한 얼굴인식 프로그램은 데이터베이스 내외부의 얼굴을 인식하는 기능을 갖추고 있다.데이터베이스에 이사람의 얼굴이 있으면 해당 사람의 이름이 표시되고 데이터베이스에 얼굴이 없으면 시스템에 'UnKnown'이 표시된다.
***


### 흐름도1
![image](https://github.com/sjy072812138/Capstone-Design-2/blob/main/IMG/%E5%9B%BE%E7%89%871.png)
### 흐름도2
![image](https://github.com/sjy072812138/Capstone-Design-2/blob/main/IMG/%E5%9B%BE%E7%89%872.png)
### GUI
![image](https://github.com/sjy072812138/Capstone-Design-2/blob/main/IMG/%E5%9B%BE%E7%89%873.png)
### UnKnown화면
![image](https://github.com/sjy072812138/Capstone-Design-2/blob/main/IMG/%E5%9B%BE%E7%89%874.png)
### 인식 화면
![image](https://github.com/sjy072812138/Capstone-Design-2/blob/main/IMG/%E5%9B%BE%E7%89%875.png)
### 적장된 DB 화면
![image](https://github.com/sjy072812138/Capstone-Design-2/blob/main/IMG/%E5%9B%BE%E7%89%876.png)

***
### 참고문헌
[1] Sophanyouly T . 基于ShuffleNet的人脸识别. 浙江大学, 2019.
[2]刘柏华. 一种基于深度残差网络的人脸识别方法:, CN110263768A[P]. 2019.
[3]谢鑫. 基于ResNet50的人脸识别模型[J]. 科技资讯, 2020.
[4] Zhang X ,  Zhou X ,  Lin M , et al. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices:, 10.48550/arXiv.1707.01083[P]. 2017.
[5] Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. [J] . Kaipeng Zhang,Zhanpeng Zhang,Zhifeng Li,Yu Qiao 0001.  IEEE Signal Process. Lett. . 2016 (10)
[6] 基于Python的深度学习人脸识别方法[J]. 薛同来,赵冬晖,张华方,郭玉,刘旭春.  工业控制计算机. 2019(02)

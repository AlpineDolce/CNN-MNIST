## 📖프로젝트 요약

![cnn.jpeg](attachment:ceb53da1-0f88-4262-b90c-8129ec789f6a:cnn.jpeg)

- 프로젝트 소개
    - PyTorch를 활용여하 CNN의 모델을 구현하여 MNIST 데이터로 해당 모델 학습하고 MNIST의 숫자 손글씨 데이터 예측하여 정확성을 개선하도록 합니다.

## 🙋🏼‍♂️기술 스택

- **프로그래밍 언어**: Python
- **모델링 및 분석 도구**: PyTorch, Scikit-learn, Numpy
- **데이터 시각화**: Matplotlib, Seaborn
- **데이터 소스**: TorchVision(MNIST)

## 📝상세 진행 과정

- **데이터 수집**
    - TorchVision 라이브러리를 통한 MNIST 수집
- **분석 계획 수립**
    - PyTorch.nn 라이브러리를 활용한 CNN 모델 코드 구현
- **모델 개발 및 성능 평가**
    
    ![output.png](attachment:52838a26-e517-4e23-89a8-371f65545653:output.png)
    
    - MNIST 손글씨 정확도 예측: CNN 모델(0.998의 정확성).
        
        ![accuracy.png](attachment:75beae82-e96e-49fb-a9cb-77d5c0f41800:accuracy.png)
        
    - 정확도 분석: Confusion Matrix로 예측 상황 파악
        
        ![cm.png](attachment:64b0f7e5-9a71-48e8-b8bd-28cb4cba6e94:cm.png)
        

## 😖향후 계획

![001.ipynb - .cursor-tutor - Cursor 2025-03-29 오후 10_41_16.png](attachment:caefce75-19ab-4059-9c0f-c2395aaed070:001.ipynb_-_.cursor-tutor_-_Cursor_2025-03-29_오후_10_41_16.png)

- **추가 개선 가능성**

- 정확도가 안정적으로 0.999 이상의 수치를 달성 도전
- 해당 모델을 활용하여 숫자 외에도 그림 등도 인식 예측 가능한 모델 만들어 챗봇 프로젝트 연계
- OPEN CV를 활용하여 사용자가 직접 작성한 숫자를 영상을 통해 숫자 예측 인식 진행
- 금융 분야의 사용자 서명 인식 정확도 프로젝트도 고려

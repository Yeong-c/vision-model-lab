### models들이 backbone 역할만 하도록 최소한으로 구현하고, methods가 그 backbone을 감싸서(projection head 등) Loss를 뱉도록 구현하면 됨.
- models = backbone (입력 -> feature)
- methods = wrapper (backbone을 받아 projection/contrastive/loss를 수행)

## 최소 인터페이스
- Backbone (models/*)
  - __init__: get_model로 Model 클래스 리턴
  - forward: feature을 flatten해서 리턴 (이건 추후에 수정해야 할 수도, 일단 이렇게 구현)

- Method (methods/*)
  - forward: Model(위 Backbone을 받아 온)에 Head 등을 붙여서 Loss를 리턴
  - predict: 필요시 Output만을 리턴하는 함수 (supervised 참고)


- Model을 추가하려고 하면 ResNet 참고
- Method를 추가하려고 하면 Supervised, SimCLR 참고

가장 기본적으로 구현한 것들이라 따라가면 쉬울 것임.

## 이식 예시
- 기존 메소드가 backbone과 head를 섞어두었다면 head부분만 methods로 옮기고 backbone은 입력->feature만 반환하도록 리팩토링.
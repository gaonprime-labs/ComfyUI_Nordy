# ComfyUI_Nordy

## 🚨 중요 공지

이 저장소는 **Public Repository**입니다.

- **원활한 서브모듈 등록**을 위해 퍼블릭으로 생성됨
- **절대로 예민한 데이터(API 키, 비밀번호, 개인정보 등)를 포함하지 마세요!**
- 추후 프라이빗으로 전환할 수 있음

## 설명

ComfyUI용 커스텀 노드 패키지

## 등록된 클래스

### SaveImageS3PresignedUrlNordy

AWS S3 presigned URL을 사용하여 이미지를 직접 업로드하는 노드

## 설치

ComfyUI의 `custom_nodes` 디렉토리에 서브모듈로 추가:

```bash
cd ComfyUI/custom_nodes
git submodule add https://github.com/yourname/ComfyUI_Nordy.git
```

## 사용법

ComfyUI 재시작 후 `Nordy` 카테고리에서 노드 사용 가능

## 주의사항

- 퍼블릭 저장소이므로 민감한 정보 절대 커밋 금지
- presigned URL은 임시 URL이므로 보안상 안전함

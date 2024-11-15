## 백엔드 실행 방법
```shell 
cd back  
npm run start:dev
```
---

## swagger 접속방법
브라우저에 http://localhost:5001/api-docs 입력

---

## docker 사용방법
1. docker 설치하기 [docker 공식페이지](https://www.docker.com/products/docker-desktop/)
2. 레포지토리의 루트 디렉토리에서 아래 명령어 실행하여 아미지 빌드
```
docker build -t ggu_back .
```
3. 생성된 이미지 확인
```
docker images
```
4. 이미지로 컨테이너 띄우기
```
docker run -d -p 3002:5002 ggu_back
```
5. 컨테이너 동작 확인
```
docker ps
```
6. 브라우저에서 아래 주소 입력해보기
```
http://127.0.0.1:3002
```


---

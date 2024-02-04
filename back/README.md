## 백엔드 실행 방법
```shell 
cd back  
npm run start:dev
```
---

## swagger 접속방법
블라우저에 http://localhost:5001/api-docs 입력

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
4. 이미지로 컨텐이너 띄우기
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


<img src="https://capsule-render.vercel.app/api?type=waving&color=BDBDC8&height=150&section=header" />

# Gugram

  **Gugram**<br/>
  구그램은 AI를 활용하여 음식 사진을 인식하여 식단을 기록 및 분석해주는 서비스입니다.


## 프로젝트 구성 안내
### **목차**<br/>
[1. 프로젝트 소개](#1-프로젝트-소개)<br/>
[2. 프로젝트 목표](#2-프로젝트-목표)<br/>
[3. 프로젝트 기능 설명](#3-프로젝트-기능-설명)<br/>
[4. 프로젝트 구성도](#4-프로젝트-구성도)<br/>
[5. 프로젝트 팀원 역할 분담](#5-프로젝트-팀원-역할-분담)<br/>
[6. 버전](#6-버전)<br/>

## 1. 프로젝트 소개

### 기술 스택 <br/>
 - AI 
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <br/>
  
 - Front-end <br/>
  typescript
   <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black"/>
   <img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=React&logoColor=black"/> <br/>

 - Back-end <br/>
   - Node.js
   - Nest.js
   - typescript
   - postgresql
   <img src="https://img.shields.io/badge/Node.js-339933?style=flat-square&logo=Node.js&logoColor=white"/>

### 라이브러리 <br/>
 - 데이터분석 <br/>
   <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
   <img src="https://img.shields.io/badge/matplotlib-0ABF53?style=flat-square&logo=multer&logoColor=white"/> <br/>

 - Front-end <br/>
   - redux
   - module css
   - 

 - Back-end <br/>
   - typeorm
   <img src="https://img.shields.io/badge/multer-FF9E0F?style=flat-square&logo=multer&logoColor=white"/>
   <img src="https://img.shields.io/badge/amazons3-569A31?style=flat-square&logo=amazons3&logoColor=white"/> <br/>
  
### AI 모델 학습 데이터셋
 AI 허브의 음식 이미지 데이터셋을 가져와 이미지 인식 모델 학습에 활용하였습니다.
 https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=74<br/>


## 2. 프로젝트 목표

  ##### 1) AI를 활용한 200가지 음식 이미지 인식 및 기록 기능

  ##### 2) Open AI API를 활용한 식단 분석 서비스 구현


## 3. 프로젝트 기능 설명

### 주요 기능 (주된 활용성) 및 서브 기능

**1. 회원가입 & 로그인**<br/>
- 이메일, 닉네임 중복 확인<br/>
- 로그인 시, token 및 cookie로 유저 인증

**2. 메인페이지**<br/>
- 목표 칼로리 대비 하루 총 섭취 칼로리 및 영양 성분별 섭취량 - 그래프 시각화
- 끼니별 대표 사진 및 섭취 칼로리

**3. 달력 페이지**<br/>
- 목표 칼로리 성취여부를 달력에 시각화하여 보여줌

**4. 하루 식단 페이지**<br/>
- 하루 섭취 칼로리는 아침, 점심, 저녁, 간식으로 나눠서 칼로리 정보 제공 
- 카드형 레이아웃

**5. 한끼 식단 페이지**<br/>
- 음식 이미지 등록 기능 ( 촬영 또는 앨범 )
    - 등록된 음식 이미지를 분석하여 얻은 음식 종류 데이터 제공
    - 여러 음식이 보이는 사진을 등록하면, 이를 분석하여 개별 음식 사진으로 등록
    - 분석된 음식명을 원본 이미지 위에 태그로 표시 </br></br>
- 음식명 검색 및 등록 기능 (사진 없음)
    - 식품의약품안전처 식품영양성분 데이터베이스 활용
    - https://various.foodsafetykorea.go.kr/nutrient/  </br></br>
- 음식 종류 별 칼로리와 영양성분 데이터 제공
  - 식품의약품안전처 식품영양성분 데이터베이스 활용
  - https://various.foodsafetykorea.go.kr/nutrient/ </br></br>

**6. AI영양사 페이지**<br/>
- 채팅형 레이아웃
- open ai api를 활용한 사용자 맞춤 식단 분석 데이터 제공 및 저장 
  - 식단 추천
  - 선택한 식단 평가
  - 목표 추천 <br/><br/>



### 프로젝트만의 차별점, 기대 효과
- AI를 활용한 음식 이미지 인식
- Open AI API를 이용한 사용자 맞춤 식단 분석 서비스 제공


## 4. 프로젝트 구성도
### 메인페이지
### 회원가입 & 로그인 & 프로필
### 습관페이지
### 캘린더 / 커뮤니티




## 5. 프로젝트 팀원 역할 분담
| 이름 | 담당 업무 |
| ------ | ------ |
| 허제인 | 팀장/프론트엔드 개발 |
| 배현진 | 프론트엔드 개발/AI 개발 |
| 양민정 | 프론트엔드 개발 |
| 오혜수 | 프론트엔드 개발 |
| 정아영 | 프론트엔드 |
| 김명지 | 백엔드 개발 |
| 장재웅 | 백엔드 개발 |
| 최유림 | 백엔드 개발 |
| 차봉준 | AI 개발 |

**멤버별 responsibility**

1. 팀장 

- 기획 단계: 구체적인 설계와 지표에 따른 프로젝트 제안서 작성
- 개발 단계: 팀원간의 일정 등 조율 + 프론트 개발
- 수정 단계: 기획, 스크럼 진행, 코치님 피드백 반영해서 수정

2. 프론트엔드 

- 기획 단계: 큰 주제에서 문제 해결 아이디어 도출, 데이터 수집, 와이어프레임 작성
- 개발 단계: 와이어프레임을 기반으로 구현, 데이터 처리 및 시각화 담당, UI 디자인 완성
- 수정 단계: 피드백 반영해서 프론트 디자인 수정

 3. 백엔드 & 데이터 담당  

- 기획 단계: 기획 데이터 분석을 통해 해결하고자 하는 문제를 정의
- 개발 단계: 웹 서버 사용자가 직접 백엔드에 저장할수 있는 기능 구현, 데이터 베이스 구축 및 API 활용, 데이터 분석 개념 총동원하기
- 수정 단계: 코치님 피드백 반영해서 분석 / 시각화 방식 수정<br/>

## 6. 버전
  - 0.0.1


<img src="ht## 백엔드 실행 방법
```shell 
cd src
npm run start:dev
```
---

## swagger 접속방법
블라우저에 http://localhost:5001/api-docs 입력

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
4. 이미지로 컨텐이너 띄우기
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

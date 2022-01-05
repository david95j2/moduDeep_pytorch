# Error log 📢

모두의 딥러닝 pytorch 영상을 수강하면서 모든 에러를 여기다가 담겠다.
<br>
크고 작은 에러들⚠️ 모두 담기!!
<br>
해결부터 해결요망까지!!
<br>
추후 내 자산💰이 되리라~~~
---
---


### 1. Visual_Studio_Cdoe 한글 받침 분리 현상
- Ubuntu 18.04.5에 설치한 VsCode에서 한글 입력 시 받침이 분리되어 출력되었다.
- "같은" 입력 시 "가ㅌ은" 으로 표시되었다.

```
1. VsCode에서 [File] -> [Preference] -> [Settings] 으로 이동하여 Font Family를 검색한다.
2. 'Droid Sans Fallback'을 삭제한다.
```

### 2. Visual_Studio_Code 빨간 밑줄 현상
- VsCode 안에서 pytorch 사용하려다 보니 compile은 문제가 없는데 빨간 밑줄이 계속해서 뜨는 상황이 발생하였다. 
- 찾아보니 python 설치할 때 Extenstion에서 Visual Studio IntelliCode 자동으로 설치되었고 이를 통해 빨간 밑줄이 발생하였다.

```
1. extension 에서 Visual Studio IntelliCode 를 uninstall 한다.
```
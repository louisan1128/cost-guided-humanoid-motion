# 코드와 데이터 출처

- 시뮬레이터: 로컬 Isaac Lab / Isaac Sim. 현재 실행 backend는 Isaac Lab이다.
- 로봇·PPO·DeepMimic·AMP: [MimicKit](https://github.com/xbpeng/MimicKit). 로컬 `/home/an/Desktop/MimicKit-main`을 외부 의존성으로 사용한다. 해당 snapshot의 정확한 commit은 아직 식별하지 못했다.
- 모션: MimicKit 모션과 사용자가 준비한 Mixamo 데이터. 변환·좌우 대칭·구간 추출·정지 유지 파생본이 있다. 원본과 파생본을 독립 데이터로 세지 않는다.
- 모션 파일의 재배포 허용 여부는 별도 확인이 필요하다. 현재 `data/`와 모델은 Git 추적에서 제외한다.

프로젝트에서 추가한 부분은 모션 변환·연결 도구, 환경 adapter, 명령 조건과 보상, 실행 설정, 평가 도구와 기록이다. PPO와 AMP 알고리즘 자체를 새로 구현한 것은 아니다.

코드와 문서 작성·디버깅에는 AI 도구의 도움을 받았다. 사용자는 실행 결과를 직접 관찰하고, 전환 시 발이 튀는 문제와 정지 유지 실패 등을 보고하며 실험 방향을 결정했다. 세부 구현은 소스와 실험 기록을 기준으로 설명한다.

# 속도 조건부 PPO smoke test

실행일: 2026-09-14

## 목적

학습 성능을 평가하기 전에 G1/Isaac Lab 초기화, 속도 command observation,
혼합 reward, PPO update, checkpoint 저장이 하나의 경로로 연결되는지 확인한다.

## 조건

- Source: 로컬 `/home/an/Desktop/MimicKit-main` 사본. Git metadata가 없어 commit은 식별하지 못함.
- Backend: Isaac Lab / PhysX, headless
- GPU: RTX 4060 Ti 8GB
- Environments: 1
- Samples: 64
- Command: episode별 0.8~1.2m/s uniform sample
- Agent: `ppo_g1_speed_smoke.yaml`; wiring 확인용 축소 설정

## 결과

- Isaac Lab과 G1 29-DoF model 초기화 성공
- `g1_walk.pkl` 로드 성공
- PPO model 생성: 2,807,838 parameters
- 32 samples씩 2 iteration 완료
- `log.txt`, 실행 config 사본과 `model.pt` 저장 성공
- 저장 checkpoint를 별도 test process에서 불러와 2 episodes 실행 성공
- 진단 로그 기록 확인: target speed, measured speed, speed MAE

Return과 tracking error는 64 samples의 무작위 초기 정책 결과이므로 성능 결론에 사용하지 않는다.
32-sample 진단 실행의 speed MAE도 초기 reference 상태 부근의 짧은 episode에서 측정된 순간값이므로
속도 제어 성공의 근거로 사용하지 않는다. 최종 평가는 episode 전체 transition을 집계해야 한다.
실행 종료 후 관측한 GPU 메모리 사용량은 peak VRAM이 아니므로 성능 자료로 사용하지 않는다.

## 다음 확인

- [x] 저장 checkpoint를 test mode로 불러오기
- [x] 로그에서 target/measured speed와 speed MAE 확인
- [ ] 고정 걷기 baseline 재현
- [ ] Peak VRAM과 환경 수별 처리량 측정
- [ ] 본 학습 전에 clean MimicKit commit 고정

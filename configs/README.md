# 설정

- `command/`: 지금 진행하는 명령 조건 PPO + AMP. 실행 안내는 해당 README.
- `envs/`: generic humanoid tracker, 전환 평가, 과거 G1 환경.
- `datasets/`: reference motion 목록. 실제 rollout dataset과 다르다.
- `agents/`: 기존 PPO 설정.
- `engines/`: simulator 설정.
- `project.yaml`: 전체 설계와 상태 기록. 직접 실행하는 환경 파일은 아니다.

기존 설정은 실험 재현을 위해 보존한다. 새 학습은 별도 output 이름을 사용한다.

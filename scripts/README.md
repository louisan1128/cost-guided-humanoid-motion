# 실행 도구

현재 작업은 [명령 조건 PPO](../configs/command/README.md)입니다.

- `train_command_policy.py`: 새 PPO + AMP 학습/평가 launcher.
- `view_continuous_chain.py`: 기존 tracker의 긴 연결 장면 확인.
- `view_transition_v2.py`: 개별 전환 장면 확인.
- `build_five_motion_chain_probe.py`, `build_selected_transition_chain.py`: reference 연결 실험.
- `convert_deepmimic_humanoid_motion.py`: 모션 변환 작업.
- `train_speed_expert.py`: 예전 G1 속도 조건 실험. 현재 command policy와 별개.

이전 실행 안내 원문은 `docs/archive/scripts_before_command_ppo.md`에 보관했습니다.
기존 스크립트는 서로 import하는 경우가 있어 옮기지 않았습니다. 각 스크립트의 `--help`와 관련 실험 기록을 함께 확인합니다.

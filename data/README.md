# 로컬 모션 파일

모션 파일은 GitHub에 포함하지 않는다. 사용한 원본·파생본의 배포 권한을 아직 확인하지 않았고, GitHub에는 코드와 설정만 올린다.

현재 명령 조건 PPO의 `configs/command/motions.yaml`은 프로젝트 루트 기준으로 아래 파일을 요구한다.

- `data/reference_motions/mixamo_start_xyfix.pkl`: 사용자가 확보한 Mixamo 출발 동작을 generic humanoid로 변환한 파일.
- `data/reference_motions/humanoid_walk_neck_fixed.pkl`: 걷기 모션의 목 자세를 보정한 파일.
- `data/reference_motions/humanoid_turn_left_5s_8s.pkl`: 걷기 중 좌회전 구간.
- `data/reference_motions/humanoid_turn_right_5s_8s.pkl`: 위 구간을 대칭 변환한 우회전.
- `data/reference_motions/inertial_walk_stop_500_v1_hold5s_probe.pkl`: 정지 전환에 5초 정지 유지를 더한 파생본.

변환·편집 코드는 `scripts/`에 있다. 원본 BVH와 DeepMimic 모션의 출처·변환 결과는 [출처 기록](../docs/provenance.md)과 [실험 기록](../docs/experiments/)을 참고한다. 원본 파일을 합법적으로 확보하고 같은 변환을 수행해야 전체 학습을 재현할 수 있다. 현재 저장소만 내려받아서는 동일한 모션 데이터와 checkpoint가 생기지 않는다.

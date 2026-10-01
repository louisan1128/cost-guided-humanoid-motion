# 연속 모션 전환 진단 v1

500-iteration PPO tracker가 여러 전환을 **reset 없이 한 episode**에서 추종하는지 보는 reference를 준비했다. 기존 다섯 원본 모션을 새로 이어 붙이지 않고, 선택한 전환 clip의 공통 걷기·회전 구간을 관절 자세와 속도로 맞췄다. 새 clip의 XY 위치와 heading을 맞춘 다음, 경계의 자세·속도 차이를 0.3초 동안 줄였다. 이는 접촉 또는 동역학 제약을 푸는 방법이 아니다.

- 왼쪽: 출발→걷기→좌회전→걷기→정지→5초 유지, 약 18.83초
- 오른쪽: 출발→걷기→우회전→걷기→정지→5초 유지, 약 18.82초
- 생성 코드: `scripts/build_selected_transition_chain.py`
- reference·경계 기록: `data/reference_motions/chain_start_walk_{left,right}_walk_stop_hold5s_v1.pkl` 및 동명 JSON
- 환경 설정: `configs/envs/chain_start_walk_{left,right}_walk_stop_hold5s_v1.yaml`
- 실행 모델: `outputs/ppo_transition_stop_hold5s_500_seed0/model.pt`

정적 검사에서 양쪽 파일 모두 34차원·120Hz·비반복 reference이며, 마지막 5초 프레임은 동일하다. 세 연결 지점의 첫 root step은 약 0.010/0.010/0.012m, 첫 joint step은 약 0.044/0.037/0.037rad이다. 이것은 reference 연속성 검사이고 PPO 물리 성공 증거는 아니다. 사용자의 GUI 실행 전에는 물리 검증과 시각적 자연스러움을 **미확인**으로 둔다.

```bash
cd /home/an/Desktop/project_alpha
/home/an/miniconda3/envs/issaclab/bin/python scripts/view_continuous_chain.py left
/home/an/miniconda3/envs/issaclab/bin/python scripts/view_continuous_chain.py right
```

각 명령은 한 episode만 재생한다. 계속 반복해 보고 싶으면 `--episodes 1000000`을 붙이고 `Ctrl+C`로 종료한다. 약 18.8초 이전에 reset되면 PPO가 중간에 종료한 것이므로 마지막으로 통과한 전환과 실패 시점을 기록한다. 전체를 끝내더라도 독립 초기 상태·새 목표 수행의 성공으로 간주하지 않는다.

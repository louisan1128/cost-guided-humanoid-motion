# 움직임을 유지하는 모션 전환 보정 v2

## 변경 이유와 방법

기존 지지 발 고정 bridge는 다음 clip 진행을 0.25초 멈춘 채 두 자세를 보간했다. Reference 발 점프는 줄었지만, 실제 걷기의 운동량을 반영하지 못해 기존 PPO에서는 오히려 넘어지기도 했다.

`scripts/build_inertial_transition_probe.py`는 다음 clip의 시간 진행을 유지한다. 연결 직전과 다음 clip의 자세·속도 차이를 구해 root 위치, root 회전, 각 관절의 보정값을 0.15/0.30/0.50초 동안 5차 곡선으로 0까지 줄인다. 회전은 rotation vector의 단순 선형 연결 대신 상대 회전 correction을 적용한다. 이 방법도 동역학 최적화나 접촉 제약 충족을 보장하지 않으므로 물리 평가를 반드시 병행한다.

기존 원본·checkpoint는 보존했다. AI가 구현한 연결 보정 코드 및 실험이며, 사용자는 기존 전환의 시각적 실패 사례를 제시했다. 외부 MimicKit 코드 변경 없음.

## 기존 고정 PPO 결과

15개 후보를 seed 0, 첫 프레임 고정 시작으로 각 1회 평가했다. 단순 clip 종료의 SUCC와 장시간 안정성을 구분한다.

- 걷기→좌회전: 세 보정 시간 모두 끝까지 진행. 0.5초 보정의 종료 root 오차 약 0.02m, 몸통 기울기 약 13.2°.
- 좌회전→걷기: 기존 실패에서 세 후보 모두 4.03초 clip 끝까지 진행. 0.3초 보정의 종료 몸통 기울기 약 6.9°. 다만 걷기 두 주기를 더 붙이면 4.37초에 실패.
- 우회전→걷기: 0.3초 보정에서 clip 끝 도달, 기울기 10.2°. 두 주기 추가 시 4.43초에 실패.
- 걷기→정지: 0.5초 보정에서 clip 끝 도달, 기울기 1.5°. 정지 유지가 확인된 것은 아님.
- 걷기→우회전: 자연 진입 구간과 기존 5–8초 회전 구간을 각각 시험했으나 실패.

정지 유지 첫 평가가 기존 약 4초 episode 제한 때문에 TIME으로 끝났다. 이후 `transition_inertial_*_v2_eval.yaml` 전부 episode_length=10초로 설정해 클립 끝보다 충분히 길게 평가한다. 시간 제한으로 종료한 첫 결과는 정지 유지 성공 근거로 쓰지 않는다.

후보 요약: `outputs/transition_diagnostics/inertial_candidate_results_v1.json`.
Reference 양발 경계/최대 step과 최저 높이: `outputs/transition_diagnostics/inertial_reference_audit_v1.json`.
원본이 가지고 있던 작은 발바닥 지면 침투도 남아 있으므로 시각적·물리적 자연스러움이 보장되지 않는다.

## PPO 적응 pilot

원본 다섯 모션과 수정한 여섯 전환을 함께 넣은 manifest:
`configs/datasets/humanoid_transition_inertial_v2.yaml`.
전환 선택 기록: `configs/datasets/transition_inertial_v2_selection.json`.

기존 `outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt` 가중치에서 새 학습을 시작한다. optimizer 상태·학습 step까지 복원하는 완전 resume가 아니다. 환경 4096개, iteration당 32 step, 100 iteration의 13,107,200 sample 예산. 출력은 별도 `outputs/ppo_transition_inertial_v2_pilot_seed0/`에 저장한다. 보상과 observation/action contract는 동일하다.

```bash
cd /home/an/Desktop/MimicKit-main
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode train --num_envs 4096 --visualize false --rand_seed 0 \
  --max_samples 13107200 --save_int_models true \
  --env_config /home/an/Desktop/project_alpha/configs/envs/deepmimic_humanoid_transition_inertial_v2.yaml \
  --agent_config /home/an/Desktop/project_alpha/configs/agents/ppo_humanoid_five_motion_pilot.yaml \
  --model_file /home/an/Desktop/project_alpha/outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt \
  --out_dir /home/an/Desktop/project_alpha/outputs/ppo_transition_inertial_v2_pilot_seed0
```

이미 실행한 output 디렉터리를 새 실행으로 덮어쓰지 않는다. 후속 평가 도구는 `scripts/evaluate_transition_v2_suite.py`; 여섯 전환과 원본 다섯 motion을 각각 같은 checkpoint로 검사한다. `scripts/view_transition_v2.py`는 사용자가 지정한 checkpoint로 전환 하나를 재생한다.

평가 결과가 나오기 전까지 모든 전환 해결이나 자연스러움 확보로 판단하지 않는다. 이 자료는 튜닝용으로 사용했으므로 독립 test set 결과로 주장하지 않는다.

## 100-iteration pilot 물리 결과

모델: `outputs/ppo_transition_inertial_v2_pilot_seed0/model.pt`.
평가 원자료: `outputs/transition_diagnostics/inertial_v2_pilot100_eval/`.

| 전환 | 결과 | 종료 시간 | 끝 몸통 기울기 |
|---|---|---:|---:|
| 출발→걷기 | SUCC | 6.03초 | 5.8° |
| 걷기→좌회전 | SUCC | 5.43초 | 10.9° |
| 좌회전→추가 걷기 | FAIL | 5.73초 | 28.2° |
| 걷기→우회전 | SUCC | 4.87초 | 12.3° |
| 우회전→추가 걷기 | FAIL | 5.27초 | 18.7° |
| 걷기→정지 및 2초 유지 | SUCC | 5.27초 | 6.3° |

정지 마지막 1초 평균 root XY 속도는 0.0276m/s. 원래 start, 좌회전, 우회전, stop clip은 각 1회 끝까지 수행했고 looping walk는 10초 TIME 종료(FAIL 아님)였다. 원본 걷기의 평균 root 오차는 약 0.266m이므로 완벽한 추종은 아니다.

FAIL인 회전→걷기는 종료 root 오차가 좌회전 0.817m, 우회전 0.915m이고 종료 시 비발 접촉은 없었다. FAIL을 모두 실제 지면에 넘어진 사건으로 표현하지 않는다. Pose termination으로 종료될 수도 있다.

다음 실험은 같은 11-reference 설정, 4096 env에서 pilot 가중치로 추가 200 iteration(26,214,400 samples). 별도 출력 `outputs/ppo_transition_inertial_v2_add200_seed0/`. 완료와 성공은 재평가 후 기록한다. 합계 300 iteration의 추가 학습이지만 각 실행의 optimizer/iteration은 새로 시작한다.

## 추가 학습 및 선택한 미리보기 모델

추가 200 iteration(26,214,400 samples)을 실제 수행했다. 전체 추가 학습 비용은 39,321,600 samples다. 마지막 모델은 여섯 clip 종료에 도달했으나 정지 마지막 1초 평균 속도 0.414m/s, 종료 속도 1.096m/s로 정지 유지가 퇴행했다. 최종 iteration을 무조건 채택하지 않았다.

중간 checkpoint 50, 100, 150의 정지 유지 진단 후 `outputs/ppo_transition_inertial_v2_add200_seed0/int_models/model_0000000050.pt`를 선택하여 여섯 전환과 원본 다섯 모션을 모두 평가했다. Checkpoint 번호는 0-based이므로 두 번째 실행의 51번째 update까지의 모델이다. 이것은 개발용 선택이며 독립 평가가 아니다.

| 전환 | 결과 | 끝 몸통 기울기 | 끝 root 위치 오차 |
|---|---|---:|---:|
| 출발→걷기 | SUCC | 3.2° | 0.487m |
| 걷기→좌회전 | SUCC | 9.8° | 0.120m |
| 좌회전→추가 걷기 | SUCC | 7.6° | 0.520m |
| 걷기→우회전 | SUCC | 12.5° | 0.126m |
| 우회전→추가 걷기 | SUCC | 7.5° | 0.090m |
| 걷기→정지 및 2초 유지 | SUCC | 8.7° | 0.129m |

정지 마지막 1초 평균 XY 속도는 0.0756m/s, 마지막 속도는 0.179m/s다. 정지가 완전히 고정되었다고 주장하지 않는다. 원래 start/좌회전/우회전/stop은 클립 끝까지 추종했고 원래 looping walk는 10초 TIME 종료, FAIL 없음. 특히 출발→걷기와 좌회전→걷기에 약 0.5m의 root 오차가 남아 있으므로 목표 도달 정확도와 자연스러움까지 완성된 것은 아니다.

원자료: `outputs/transition_diagnostics/inertial_v2_selected150_eval/`. 선택 모델 해시·평가 설정은 `configs/transition_preview_v2.json`에 기록했다. `scripts/view_transition_v2.py`의 기본 checkpoint는 이 모델이며 모든 전환에 동일하게 사용한다.

```bash
cd /home/an/Desktop/project_alpha
/home/an/miniconda3/envs/issaclab/bin/python scripts/view_transition_v2.py walk_stop
```

인자는 `start_walk`, `walk_left`, `left_walk`, `walk_right`, `right_walk`, `walk_stop` 중 하나다. 원본 reference가 아닌 PPO 물리 추종을 보여준다. `--episodes 1`이면 한 번 실행한다. 이 미리보기는 각 쌍을 별도 reset에서 시작하며 여섯 전환을 하나의 긴 episode로 연결한 검증은 아니다. 다양한 초기 상태와 완전 연속 체인은 후속 평가가 필요하다. 학습 및 진단 프로세스는 종료되었다.

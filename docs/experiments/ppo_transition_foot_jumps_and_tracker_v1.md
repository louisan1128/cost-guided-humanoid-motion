# 전환 시 발 튐과 PPO 실패 분리 진단

사용자 화면 관찰: 출발→걷기는 양호, 걷기→좌회전은 대체로 진행하지만 한 발이 튐, 좌회전→걷기는 발이 걸리고 다른 전환은 넘어짐. 이 문서는 **reference 목표 자체의 발 이동**과 **PPO가 실제로 움직인 발**을 구분한다. 한 고정 초기 상태(seed 0), 기존 다섯 모션 PPO checkpoint, Isaac Lab 30Hz 물리 제어의 진단이며 성공률은 아니다.

## 발 reference에 실제 튐이 있는가?

`MJCFCharModel` FK로 humanoid.xml의 양발바닥 위치를 120Hz reference에서 계산했다. 표는 연결 경계 **한 프레임**의 왼발·오른발 이동량이다. 원본 정지 clip 내부의 발 최대 이동은 약 0.025m/프레임으로, 걷기→정지의 큰 점프는 clip 내부가 아니라 연결에서 발생한다.

| 연결 reference | 왼발 / 오른발 이동 |
|---|---:|
| 출발→걷기 | 0.150 / 0.095m |
| 걷기→좌회전 자연 진입 | 0.035 / 0.021m |
| 좌회전→걷기 자연 이탈 후보 | 0.053 / 0.046m |
| 걷기→우회전 자연 진입 후보 | 0.033 / 0.056m |
| 우회전→걷기 8초 연결 | 0.069 / 0.046m |
| 걷기→정지 지지 발 정렬 | 0.077 / 0.062m |

출발→걷기는 수치상 튐이 가장 커도 PPO가 클립 끝까지 추종한다. 따라서 **발 위치 경계 차이 하나만으로 물리 실패를 설명할 수 없다.** 걷기→좌회전에서는 기준 지지 왼발의 경계 이동이 작고 실제 지면 반력도 유지됐다. 실제 오른발 속도는 연결 직후 최대 약 4.1m/s였으며 그 순간 오른발은 공중에 있었다. 화면의 발 튐은 기준 reference의 큰 순간이동보다는 swing foot에 대한 정책 반응일 가능성이 높다. 이는 시계열로 확인한 가설이지 시각적인 자연스러움의 자동 판정은 아니다.

좌회전→걷기 reference의 경계 주변에서 기준 발바닥이 지면 아래 1cm 이상 침투한 프레임은 없었다. 따라서 사용자가 본 발 걸림을 **reference의 뚜렷한 지면 관통**으로 설명할 근거는 없다. 실제 tracker는 연결 약 0.4초 뒤 넘어졌다. 현재 상태와 새 걷기 목표의 불일치 또는 이 전환 경험 부족을 우선 의심한다.

## 지지 발 고정 bridge 실험

`scripts/smooth_contact_transition_probe.py`로 경계에 0.25초 구간을 삽입했다. 대상 모션의 root를 지지 발 위치에 맞춰 이동하고, root·관절을 보간하며 bridge 동안 지지 발바닥을 고정한다. 원본 다섯 클립과 기존 probe는 덮어쓰지 않았다. 이 작업은 **기구학적 보정**이며 동역학적 실행 가능성을 보장하지 않는다.

| 전환 | 이전 발 이동 최대 → bridge 내부 최대 | 고정 PPO, 보정 전 → 보정 후 |
|---|---:|---|
| 걷기→정지 | 0.077 → 0.0027m/프레임 | FAIL 2.00초 → FAIL 2.17초 |
| 걷기→좌회전 | 0.035 → 0.002m/프레임 | SUCC 5.43초 → **FAIL 4.00초** |
| 좌회전→걷기 | 0.053 → 0.001m/프레임 | FAIL 3.97초 → FAIL 4.57초 |
| 걷기→우회전 | 0.056 → 0.002m/프레임 | FAIL 3.87초 → FAIL 3.40초 |
| 우회전→걷기 | 0.069 → 0.002m/프레임 | SUCC 4.03초 → SUCC 4.30초, 종료 몸통 기울기 57.4° |

정지 bridge를 0.50초와 0.75초로 늘려도 각각 1.97초에 FAIL했다. 부드러운 목표가 곧 안정적인 제어는 아니라는 직접 증거다. **기존에 물리적으로 더 나았던 좌회전 원본을 bridge 버전으로 교체하지 않는다.** 우회전→걷기도 종료 코드만 SUCC이고 기울기 57.4°라 안정적 성공으로 보지 않는다. `outputs/transition_diagnostics/`에 보정 전후 CSV·JSON을 보존했다.

## 다음 실험: 전환을 포함한 PPO tracker 적응

현재 PPO는 원래 다섯 clip을 독립적으로 학습했고, 각 clip 사이의 물리 상태에서 새 reference를 추종하는 경험은 없었다. `configs/datasets/humanoid_five_motion_transitions_pilot_v1.yaml`은 원본 다섯 clip을 그대로 두고, 출발→걷기와 선택한 전환 probe 5개를 별도로 추가한다. `configs/envs/deepmimic_humanoid_five_motion_transitions_pilot_v1.yaml`은 같은 observation/action/reward를 사용한다. 11개 모션은 MimicKit MotionLib로 로드되는 것을 확인했다. 기존 checkpoint와 새 환경의 단일 headless 평가도 로드·실행되었다. 이 1회 평가는 무작위로 선택된 clip 하나의 호환성 점검이며 전환 성능 검증은 아니다.

이 manifest는 **실험 후보**다. 일부 전환 reference는 물리적으로 추종되지 않았으므로 성공을 보장하지 않는다. 먼저 별도 출력 디렉터리에서 짧은 adaptation pilot을 수행하고, 기존 다섯 모션의 성능이 떨어지는지도 함께 평가해야 한다. 기존 3000-iteration checkpoint는 보존한다. `--model_file`은 가중치 초기화에 쓰되 변경된 모션 분포의 새 학습으로 취급한다. iteration과 samples를 기존 실험과 합산해 완전한 resume로 설명하지 않는다.

사용자가 실행할 100-iteration pilot 명령(4096 env × 32 step × 100):

```bash
cd /home/an/Desktop/MimicKit-main
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode train --num_envs 4096 --visualize false --rand_seed 0 \
  --max_samples 13107200 --save_int_models true \
  --env_config /home/an/Desktop/project_alpha/configs/envs/deepmimic_humanoid_five_motion_transitions_pilot_v1.yaml \
  --agent_config /home/an/Desktop/project_alpha/configs/agents/ppo_humanoid_five_motion_pilot.yaml \
  --model_file /home/an/Desktop/project_alpha/outputs/ppo_humanoid_five_motion_continue_2500_seed0/model.pt \
  --out_dir /home/an/Desktop/project_alpha/outputs/ppo_humanoid_transition_adapt_pilot_seed0
```

이 학습은 아직 실행하지 않았다. Pilot 후 같은 6개 전환과 원본 5개를 전부 재평가하고, 실제 발 접촉·몸통 기울기·종료 후 정지 유지까지 확인해야 한다. 실패하면 더 긴 학습을 무조건 진행하기 전에 reference 동역학과 접촉 제약을 재검토한다. Diffusion 학습은 이 관문 이후다.

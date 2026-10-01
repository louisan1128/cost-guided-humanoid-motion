# 최종 프로젝트 계획 보고서

**Goal-Conditioned Motion Diffusion for Closed-Loop Humanoid Control**

**목표 조건에 맞는 전신 모션 생성과 RL 추종: 생성 품질과 실제 실행 성능의 차이 분석**

> 2026-09-22 최종 설계 v1.0. 연구 질문과 필수 범위를 고정한다.
> 2026-10-01 상태: generic humanoid tracker와 전환·정지 유지 실험을 진행했다. 명령 조건 PPO + AMP를 추가했으며, 생성기 비교는 아직 미구현이다. 현재 진행은 [진행 기록](progress.md)을 기준으로 한다.
> Project 1을 먼저 완결한 뒤 Project 2와 Project 3으로 확장한다. 후속 계획은 현재 필수 범위의 완료 기준을 바꾸지 않는다.

## 1. 한 문장 목표

**기존 모션과 강화학습으로 얻은 운동 경험을 이용해 목표 위치·도착 방향·정지 조건에 맞는 미래 전신 모션을 생성하고, 하나의 고정된 RL tracker로 실행했을 때의 목표 달성 능력과 실패 원인을 검증한다.**

대표 데모는 평지에서 가까운 목표를 지정하면 휴머노이드가 방향을 바꾸고 이동한 뒤 목표 방향을 바라보며 멈추는 것이다. 시작 상태는 정지 또는 보행 중이다. 목표 거리·각도·허용 시간은 실제 데이터와 tracker의 검증 범위에서 정한다. 처음부터 임의의 방향, 급회전, 높은 속도를 요구하지 않는다.

Stand·walk·turn은 필요한 운동 범주다. 정확히 네 개의 파일로 제한하지 않는다. 정지 자세와 감속·정지 전환은 별개의 데이터다. Run, 장애물 회피, 지형 보행은 이번 필수 범위가 아니다.

## 2. 확정한 연구 질문과 가설

> **동일한 운동 데이터와 고정된 RL tracker를 사용할 때, 목표 조건부 Motion Diffusion은 모션 검색·연결 및 MLP와 비교해 새로운 목표 조합을 얼마나 정확하고 안정적으로 수행하는가? 실제 상태 피드백과 재계획 간격은 생성 모션과 물리 실행 사이의 오차에 어떤 영향을 주는가?**

새로운 목표 조합은 학습과 분리한 상대 위치·도착 방향·도착 시간의 조합을 뜻한다. 새로운 운동 기술 습득, 무제한 외삽 또는 모든 task에 대한 zero-shot 제어를 뜻하지 않는다.

| 검증할 가설 | 확인 방법 |
|---|---|
| H1. 낮은 offline 모션 오차만으로 목표 달성을 예측하기 어렵다 | 동일 시나리오에서 reference의 목표 오차와 실제 목표 오차·넘어짐을 함께 측정 |
| H2. 실행 이력 피드백이 예측 이력 재사용보다 오차 누적을 줄인다 | 나머지 입력·tracker를 유지한 채 이력 채널만 교체 |
| H3. 실행 chunk 길이에 안정성·반응성·계산 비용의 trade-off가 있다 | 동일 생성 모델의 실행 길이만 세 수준으로 변경 |

가설과 반대되는 결과도 공개한다. Diffusion의 우월성은 전제가 아니다. MLP가 더 정확하거나 빠를 수 있다. 데이터가 거의 단일한 해만 제공한다면 diffusion의 다중 해 표현이 이점이 아닐 수 있다. 그 사실 자체가 모델 선택에 유용한 결과다.

## 3. 실용적 의미와 주장 한계

장기 관심은 운동 경험을 재사용하여 새로운 요구에 대응하는 것이다. 이번 프로젝트는 목표 조건이 바뀌어도 최종 생성기와 tracker의 가중치를 고정한 상태에서 실행하는 인터페이스를 만든다. 사람이 발의 순서와 전신 관절 궤적을 직접 지정하지 않고 목표를 제시한다.

단, 목표 조건부 PPO 역시 새로운 목표에 대응할 수 있다. 본 프로젝트가 PPO보다 개발 노력을 줄이거나 실용적으로 우월하다고 주장하지 않는다. 직접 task PPO와 비용을 통제한 비교는 필수 범위에 없으므로, 결론은 세 모션 생성 방식과 그 실행 특성에 한정한다.

데이터 준비, RL 학습, 튜닝, 생성 모델 학습과 추론 비용을 모두 기록한다. Generic humanoid의 시뮬레이션 결과를 G1이나 실제 로봇 배포 성과로 표현하지 않는다. 포트폴리오의 기여는 재현 가능한 구현, 명확한 비교, 실패 원인 분석에 둔다.

## 4. 논문에서 가져올 것과 가져오지 않을 것

| 연구 | 확인한 핵심 | 이번 설계에 반영 |
|---|---|---|
| [CLoSD, ICLR 2025](https://xbpeng.github.io/projects/CLoSD/CLoSD_2025.pdf) | 생성기와 RL tracker의 closed-loop 연결, tracker 추가 학습 | 생성/실행 이력 구분과 tracker 일반화 검증. 텍스트·물체 상호작용 전체 재현은 제외 |
| [ARDY, SIGGRAPH 2026](https://research.nvidia.com/labs/sil/projects/ardy/) | waypoint·경로·자세 조건의 온라인 모션 생성, SONIC과 연결 | 목표 조건을 명시하는 표현. 대규모 모델이나 G1 checkpoint 호환성을 가정하지 않음 |
| [BeyondMimic, 2025 공개 논문](https://arxiv.org/html/2508.08241v1) | 상태·action 공동 생성과 실행 시 비용 guidance | 경험 재사용과 학습 비용 구분. 현재 reference→tracker 구조와 동일시하지 않음 |
| [Diffuse-CLoC, 2025](https://arxiv.org/abs/2503.11801) | 상태·action 공동 모델로 제어 가능한 생성, 계층형 제어와 비교 | 생성의 목표 정확도와 실제 실행의 차이를 주요 문제로 설정 |
| [PDP, SIGGRAPH Asia 2024](https://tml.stanford.edu/PDP.github.io/) | RL 교정 경험으로 직접 diffusion action policy의 오차 누적에 대응 | 완벽한 모션만으로 배포 상태를 덮을 수 없다는 점을 반영. action label 방식을 그대로 복제하지 않음 |
| [SMP, SIGGRAPH 2026](https://yxmu.foo/smp-page/) | diffusion을 재사용 가능한 RL 모션 보상으로 활용 | 대안의 역할을 구분. 본 프로젝트는 온라인 reference 생성으로 고정 |
| [Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking, 2026 공개 논문](https://arxiv.org/abs/2604.17335) | 지형 조건의 생성, RL tracking, closed-loop tracker 추가 학습 | 지형 확장은 새 데이터·관측·추종 검증이 필요한 별도 과제로 둠 |

이 연구들의 모델 크기·GPU 예산·로봇 성능을 그대로 재현한다고 약속하지 않는다. 작은 모션 범위에서 목표 조건과 생성–실행 오차를 분석하는 실증 프로젝트다. [상세 연구 근거](related_work.md)

## 5. 최종 시스템과 RL의 역할

### 준비·학습

```text
공개 walk·turn·start/stop·stand 모션 검증
                   ↓
MimicKit의 reference-conditioned tracker를 PPO로 직접 학습
                   ↓
Reference 추종 rollout 수집: 실제 전신 상태·시간·원본 reference
                   ↓
실제 미래 상태로 목표 위치·방향·시간·속도 label 구성
                   ↓
동일 데이터로 검색 library / MLP / Motion Diffusion 구성
```

### 실행

```text
실제 현재 상태·최근 실행 이력
+ 목표 위치·방향·남은 시간·도착 속도
                   ↓
검색·연결 / MLP / Motion Diffusion
                   ↓
짧은 미래 전신 reference: 관절 자세 + root 이동·회전
                   ↓
공통 reference adapter → 고정된 PPO tracker → 물리 시뮬레이션
                   ↓
실제 실행 상태를 관측하고 재생성
```

RL은 사라지지 않는다. 사용자가 PPO로 tracker를 직접 학습·검증한다. Tracker는 원본 모션을 실행해 데이터를 만드는 역할과 생성 모션을 실행하는 역할을 같은 reference 인터페이스로 수행한다. **2026-10-01 사용자 요청에 따라 명령 조건 PPO + AMP를 별도 데이터 수집 후보로 추가한다.** 에피소드 도중 `(vx, wz)`를 바꿔 전환을 학습한다. 이 정책은 reference tracker와 다른 모델이다. 먼저 제한된 명령 범위를 검증하고, rollout을 기존 tracker가 재실행할 수 있는지 확인한 뒤 생성기 데이터로 채택한다. 실패하면 수동 연결 데이터와 비교하여 원인을 기록하며 Diffusion이 자동으로 고친다고 가정하지 않는다.

기존 G1 속도 정책과 새 generic humanoid tracker는 입력·asset·목적이 달라 checkpoint를 그대로 옮기지 않는다. MimicKit의 multi-motion 추종 지원은 출발점이며, 외부에서 새 reference를 넣는 adapter와 재계획 처리는 직접 구현·검증할 부분이다. [공식 DeepMimic 안내](https://github.com/xbpeng/MimicKit/blob/main/docs/README_DeepMimic.md)

최종 평가 중 tracker를 동결한다. 개발 중 보완 학습이 필요하면 train 데이터와 공통 reference 변형만 사용하고, 새 버전을 만든 뒤 모든 생성 방법을 다시 평가한다. Diffusion 전용으로 튜닝한 tracker를 공통 기준으로 사용하지 않는다.

## 6. 고정 범위와 기술 선택

| 항목 | 결정 |
|---|---|
| 모델 | MimicKit generic humanoid, `data/assets/humanoid/humanoid.xml` |
| 시뮬레이터 | 기존 Isaac Lab backend를 우선 검증하고 하나로 고정. 버전은 smoke 후 lock |
| 공간·관측 | 장애물 없는 평지, 관절·몸체 벡터 상태. 영상·자연어 없음 |
| 핵심 동작 | 걷기, 완만한 좌우 선회, 출발, 감속·정지, 서 있기 |
| 최종 데모 | 가까운 목표 위치에 도달해 지정 방향을 바라보고 정지 유지 |
| 생성기 | 작은 state/goal-conditioned temporal diffusion. 기존 DDPM/DDIM 계열 구현 재사용 |
| 기준선 | 모션 검색·연결, 동일 chunk를 출력하는 deterministic MLP |
| 출력 | 미래 전신 reference. 별도 상위 trajectory diffusion이나 직접 action diffusion 없음 |
| 피드백 | 실제 상태에서 짧게 실행하고 재계획. 기본 diffusion은 후보 1개 |
| 자원 | 단일 GPU, tracker와 생성 모델을 순차 학습. 4096 env나 대형 모델을 요구하지 않음 |

속도 범위는 고정된 0–2 m/s가 아니라 reference와 tracker에서 확인한 보행 범위다. 제자리 회전·후진·급격한 방향 반전은 필수가 아니다. 목표 yaw도 데이터에서 관측된 범위로 제한한다.

Pilot에서는 약 1초 미래 reference와 그보다 짧은 실행 구간으로 시작한다. History 길이, 목표까지 시간 범위, 제어 주기, DDIM step 수와 실행 길이는 validation으로 고정한다. 이 수치는 논문 성능을 보장하는 설정이 아니다.

## 7. 데이터 전략과 현재 확보 상태

| 항목 | 2026-09-22 확인 범위 | 다음 조치 |
|---|---|---|
| Generic humanoid asset | 로컬 XML 존재 | Isaac Lab 재생·물리 실행 확인 |
| 직진 walk | `humanoid_walk.pkl` 존재, frame 34차원 | 단위·관절 순서·발 접촉 확인. 34는 action 차원이 아님 |
| 긴 walk | `long/humanoid_long_walk*.pkl` 존재 | 궤적·몸 방향과 영상으로 선회 구간 확인. 이름만으로 좌우 turn 확정 금지 |
| Run left/right | SIE 파일 존재 | 걷기 회전 데이터와 구분. 기본 dataset에는 넣지 않음 |
| 출발·정지 | 현재 MimicKit에서 사용 가능한 전환 clip 미검증 | 원본 DeepMimic `amass/humanoid3d_ready_to_walk_mirror.txt`, `humanoid3d_walk_to_ready_mirror.txt`를 후보로 검증·변환 |
| Stand | 안정적인 standalone reference 미검증 | 유효한 정지 구간 우선 확보. 자세 반복만으로 균형 검증을 대체하지 않음 |
| Locomotion dataset YAML | `MimicKit-main/data/datasets/dataset_humanoid_locomotion.yaml` 없음 | 전체 AMP 목록 복원보다 프로젝트용 작은 manifest 구성 |

출발·정지 후보는 파일 존재만 확인했으며, 원하는 동작인지·접촉이 적절한지·변환 가능한지는 미검증이다. 원본 TXT와 MimicKit PKL은 관절 표현·좌표 변환이 필요할 수 있다. 이전 설명의 프레임 수 일치만으로 동일 모션·정확한 변환을 증명하지 않는다. 출처·라이선스·변환·재생을 기록한다.

실제 감속·정지와 재출발 구간을 확보하지 못하면 정지 기능 완료로 진행하지 않는다. 약 1–2주의 pilot 예산 내에 해결 여부를 보고한다. 불가능하면 불완전한 결과로 명시하고 범위 변경은 사용자와 결정한다.

목표 label은 rollout의 실제 미래 위치·방향·시간으로 만든다(hindsight relabeling). 배포 때도 사용자가 지정하는 목표 조건을 학습에 만드는 절차이며, 미래 관절 자세 등 목표 외 정보는 입력하지 않는다. 정지 label은 낮은 선속도·각속도가 일정 시간 유지된 구간에만 붙인다. 매 걷기 window의 마지막을 정지로 취급하지 않는다.

원본 모션 그룹과 초기 조건·목표 조합은 window 생성 전에 분할한다. 같은 clip의 mirror·crop·반복 rollout은 같은 split에 둔다. Source가 너무 적으면 ‘새 모션 일반화’ 대신 ‘같은 운동 범위 안의 새 목표·초기 조건’으로 주장 범위를 축소하고 공개한다. [Dataset 규약](dataset_spec.md)

## 8. 필수 실험

| 실험 | 비교·목적 |
|---|---|
| 기반 검증 | Reference 재생과 물리 tracking 분리. 정지·선회·전환별 실행 성공률 확인 |
| 주 비교 | 검색·MLP·Diffusion을 동일 목표·초기 상태·tracker·adapter로 실행 |
| 피드백 ablation | 실제 실행 이력 대 이전 예측 이력. 현재 상태와 tracker의 관측은 유지 |
| 실행 길이 ablation | 고정 prediction horizon에서 짧음·중간·김. 호출 수·지연과 성능 함께 측정 |
| 실패 진단 | 저장된 reference, 생성 reference, 실제 실행 비교. 동일 초기 상태 replay 확인 |

평가는 먼저 분포 내 새 episode에서 하고, 이후 사전에 제외한 목표 위치–방향 조합에서 한다. 목표 도착 후 정지 유지, 도달 전 넘어짐, 시간 초과를 구분한다. 약한 외력 한 종류는 여력이 있을 때 진단 실험으로만 추가한다.

주요 결과는 목표 도달·정지 성공률, 위치·방향 오차, 잔여 속도, 넘어짐, 완료 시간, 발 미끄러짐, reference 추종 오차, 추론 지연이다. Offline 오차와 실제 성공률을 연결한다. MLP와 diffusion의 목표 정보·모션 표현·학습 데이터·horizon은 동일하게 맞춘다. [상세 평가](evaluation.md)

Held-out 실제 reference를 tracker에 넣는 실험은 미래 정답을 제공하는 진단용 기준이다. 공정한 생성기 순위나 이론적 상한으로 취급하지 않는다. 본 비교는 실용적인 PPO 대체를 입증하지 않는다.

## 9. 진행 순서와 중단 관문

| 단계 | 할 일 | 산출물·통과 근거 |
|---|---|---|
| 0. 환경·데이터 audit | Generic humanoid, walk·선회·start/stop 후보 검증 | 버전·source manifest, 재생 영상, 품질표 |
| 1. RL tracker | 기존 DeepMimic/PPO로 작은 모션 집합 추종 | 유형별 tracking 성공률·오차·영상, 외부 reference 입력 확인 |
| 2. Dataset pilot | 실제 rollout 저장, 목표 label과 split 검증 | 정지 label 검증, window 재생, train-only 통계 |
| 3. 검색·MLP 실행 | 목표 조건→reference→tracker 전체 연결 | 목표 이동·정지의 첫 closed-loop 결과 |
| 4. Motion Diffusion | 같은 데이터·인터페이스에서 학습·실행 | 생성 및 물리 실행 결과, 가중치·설정 고정 |
| 5. 비교·분석 | 주 비교, feedback와 실행 길이 ablation | 결과표·실패 사례·지연·비용 |
| 6. 문서화 | 재현 안내·영상·보고서·지원 요약 | 실행 순서와 결론을 확인할 수 있는 저장소 |

단일 reference도 추종하지 못하면 생성 모델로 넘어가지 않는다. 작은 train subset도 학습하지 못하면 데이터·좌표부터 확인한다. 목표가 지원 범위 밖이면 기능 추가보다 목표 범위를 줄인다. 성공한 validation 조건만 골라 test를 바꾸지 않는다.

12주 배분 초안: audit 2주, tracker 3주, dataset·기준선 2주, diffusion 2주, 비교·보고 3주. 일정·GPU 비용은 보장치가 아니다. Pilot 후 처리량과 학습량을 기록해 배분을 조정하되 마지막 비교·보고 기간을 기능 추가로 없애지 않는다. 여러 RL 시스템이나 논문 전체 재현을 병행하지 않는다.

## 10. 완료 기준과 산출물

- [ ] Generic humanoid와 simulator 하나에서 PPO tracker를 직접 학습·검증했다.
- [ ] 걷기·선회·출발·정지 데이터의 출처·전환·split을 검증했다.
- [ ] 목표 조건으로 전신 모션을 생성하고 실제 상태 피드백으로 실행했다.
- [ ] 검색·MLP·Diffusion을 같은 tracker에서 비교했다.
- [ ] Feedback·실행 길이 실험과 생성–실행 실패 분석을 완료했다.
- [ ] 학습·수집·추론 비용과 지원 가능한 목표 범위를 공개했다.
- [ ] 재현 저장소, 동일 조건 비교 영상, 결과 보고서, 지원용 1페이지 요약을 작성했다.

시스템 데모 성공과 연구 평가 완료를 구분한다. 성공률의 구체 임계값은 pilot 후 본 평가 전에 고정한다. 목표를 안정적으로 수행하지 못했다면 분석을 마쳐도 ‘제어 시스템 완성’으로 표현하지 않는다. Diffusion의 우월성이나 논문 수준 novelty는 완료 조건이 아니다.

## 11. Project 1에서 Project 3까지의 확장 로드맵

세 프로젝트는 하나의 시스템을 무리하게 한 번에 구현하는 계획이 아니다. **Project 1을 독립된 메인 포트폴리오로 완결하고, 검증된 데이터·tracker·평가 코드를 재사용해 조건의 복잡도를 한 단계씩 높이는 순서**다.

```text
Project 1  Goal-Conditioned Motion Diffusion
           가까운 도착 위치·방향·시간·정지 조건
                         ↓
Project 2  Trajectory-Conditioned Motion Diffusion
           중간 waypoint·root 경로·속도 조건
                         ↓
Project 3  Obstacle-Aware Navigation
           장애물 기하와 충돌 비용을 고려한 경로 실행
```

### Project 1 — Goal-Conditioned Motion Diffusion

현재 보고서의 필수 범위다. 실제 현재 상태와 최근 이력, 가까운 목표 위치·도착 방향·남은 시간·도착 속도를 입력받아 짧은 미래 전신 reference를 생성한다. 검색·MLP·Diffusion을 하나의 PPO tracker로 실행하여 목표 달성, 정지, 넘어짐, 지연과 생성–실행 오차를 비교한다.

핵심 질문은 **endpoint 조건만 주어졌을 때 목표 조건부 생성기가 데이터 안의 운동 경험을 조합해 새로운 목표 조합을 물리적으로 수행할 수 있는가**이다. Project 1의 데이터·비교·실패 분석·보고서가 완료되기 전에는 Project 2 구현을 시작하지 않는다.

### Project 2 — Trajectory-Conditioned Motion Diffusion

Project 1의 endpoint 조건에 **시간이 붙은 중간 waypoint 또는 짧은 root trajectory**를 추가한다. 생성기는 주어진 경로와 속도 변화를 따르는 미래 전신 reference를 출력하고, 가능하면 Project 1에서 검증한 tracker와 adapter를 그대로 사용한다. 필요한 경로 label은 기존 물리 rollout의 실제 root 궤적에서 우선 만들고, 지원 범위가 부족할 때만 데이터를 추가한다.

핵심 질문은 **도착점만 조건으로 줄 때보다 경로 조건을 명시했을 때 경로 추종과 회전·감속 제어가 얼마나 좋아지며, 계산 비용과 물리 안정성은 어떻게 달라지는가**이다. 비교 대상은 endpoint-only 모델, trajectory-conditioned MLP, trajectory-conditioned Diffusion이다. 같은 경로·초기 상태·tracker·실행 길이와 collision-free 평지를 사용한다.

Project 2의 통과 조건은 보지 않은 완만한 곡선과 waypoint 조합에서 실제 root 경로 오차를 측정하고, 생성 reference와 물리 실행의 차이를 설명하는 것이다. 이 단계에서는 장애물 회피를 주장하지 않는다.

### Project 3 — Obstacle-Aware Navigation

Project 2가 안정적으로 경로를 실행한 뒤, 평지에 **정적 장애물 하나부터** 추가한다. 목표와 함께 장애물 위치·크기 등의 기하 정보 또는 명시적인 충돌 비용을 생성 과정에 제공하여, 충돌하지 않는 local trajectory와 전신 reference를 만들고 같은 tracker로 실행한다.

핵심 질문은 **경로 조건부 모션 생성에 장애물 정보를 결합했을 때 목표 도달과 전신 안정성을 유지하면서 안전한 우회가 가능한가**이다. 목표 성공률·충돌률·넘어짐·최소 clearance·경로 길이·도달 시간·추론 지연을 함께 측정한다. 고정된 collision checker와 동일한 계산 예산에서 고전적 경로계획기+follower, trajectory-conditioned MLP, trajectory-conditioned Diffusion을 비교한다.

외부 경로계획기가 완성된 우회 경로를 모두 제공한다면 Diffusion의 역할은 장애물 회피 계획이 아니라 **그 경로를 실행 가능한 전신 모션으로 변환하는 것**이라고 정확히 표현한다. Diffusion의 회피 기여를 주장하려면 장애물 조건이나 충돌 비용이 생성 결과에 직접 영향을 주어야 한다. 후보 여러 개를 생성해 고르는 방식을 추가하면 검색·MLP에도 같은 충돌 검사와 선택 예산을 제공한다.

### 확장 규칙

- 각 Project의 dataset manifest, checkpoint, 설정과 결과표를 분리한다. 후속 성공으로 앞 단계의 결과를 덮어쓰지 않는다.
- Tracker를 보완하면 새 버전을 기록하고 모든 비교 방법을 같은 tracker에서 다시 평가한다. 동일한 tracker를 계속 사용할 수 있는지는 실험 결과로 판단한다.
- Run과 walk↔run은 추가 데이터와 tracker 검증이 필요한 운동 범위 확장이다. Project 2나 3의 필수 조건은 아니다.
- Terrain walking은 높이·접촉·착지 관측과 새로운 데이터·tracker 학습이 필요하므로 Project 3 이후의 별도 연구로 둔다.
- Project 3까지 자연어, vision, SLAM, 동적 장애물, 실제 로봇, G1 재이식, residual RL, state-action co-diffusion, SMP 보상, 자동 실패 증강, 새로운 diffusion 구조와 전역 navigation은 포함하지 않는다.

이 로드맵은 CLoSD의 생성기–tracker 연결, ARDY의 waypoint·경로 조건, BeyondMimic의 목표 지향 실행과 비용 사용, Diffuse-CLoC의 계층형 제어 비교에서 아이디어를 가져오되 현재 자원에 맞게 범위를 축소한 것이다. 논문 전체를 재현하거나 동일 성능을 주장하지 않는다.

**새 논문이나 데모를 볼 때마다 Project 1의 구조를 바꾸지 않는다.** 모델 크기·horizon·목표 범위는 pilot에서 조정할 수 있지만, 연구 질문·generic humanoid·reference 생성→고정 tracker·필수 비교는 유지한다. 불가피한 변경은 원인과 기존 결과를 남기고 사용자와 명시적으로 결정한다.

## 12. 바로 다음 한 단계

**명령 조건 PPO + AMP를 학습하고 고정 명령 순서에서 속도·선회·정지 유지 여부를 확인한다.** [설정과 명령어](../configs/command/README.md). 학습 완료 후 실제 rollout 수집과 공통 tracker 재실행을 검증한다. 기존 reference audit·tracker·연결 실험은 삭제하지 않는다.

관련 문서: [Dataset](dataset_spec.md) · [입출력 계약](observation_action_contract.md) · [평가](evaluation.md) · [연구 근거](related_work.md) · [출처](provenance.md) · [보고 자료](../reports/README.md)

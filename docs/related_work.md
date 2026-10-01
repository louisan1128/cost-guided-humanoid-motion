# 논문 검토와 최종 설계 근거

검토일: 2026-09-22. 기준 설계: [최종 계획 v1.0](plan.md).
저자 논문·공식 프로젝트·공식 구현 문서를 우선했다.
아래 내용은 확인한 구조와 그로부터 내린 프로젝트 판단이다. 모든 논문의 코드를 실행하거나 결과를 재현한 것은 아니다.

## 1. 역할을 구분해서 읽기

| 논문 | 확인한 자료 | 모델 역할과 이번 판단 |
|---|---|---|
| [CLoSD](https://xbpeng.github.io/projects/CLoSD/CLoSD_2025.pdf), ICLR 2025 | 본문: overview, feedback, tracker fine-tuning | 모션 reference 생성과 RL 실행의 분리. 실제 모션을 다음 생성의 prefix로 제공한다. 우리 구조의 주 참고이며, 범용 tracker가 생성 reference를 자동으로 실행한다는 가정은 피한다. |
| [ARDY](https://research.nvidia.com/labs/sil/projects/ardy/), SIGGRAPH 2026 | 공식 프로젝트의 입력·구조·로봇 응용 설명 | 온라인 생성기에 waypoint·root 경로·전신/부분 자세 조건을 준다. 대규모 데이터와 SONIC 결합 성과를 작은 MimicKit 모델로 재현한다고 가정하지 않는다. |
| [BeyondMimic](https://arxiv.org/html/2508.08241v1), 2025 공개 버전 | 본문 III–IV, 공식 프로젝트 | PPO tracking 경험을 상태·action 공동 diffusion으로 옮기고 task cost로 guidance한다. 현재의 reference→tracker 방식과 다르다. 구현 구조 대신 운동 경험 재사용이라는 문제 정의를 참고한다. |
| [Diffuse-CLoC](https://arxiv.org/abs/2503.11801), 2025 | 저자 초록 | 상태 예측과 action을 함께 모델링해 제어 가능성과 물리 실행을 연결한다. 계층형 생성–추종의 불일치를 평가할 이유를 제공한다. 이 논문의 전체 구현을 재현하지 않는다. |
| [PDP](https://tml.stanford.edu/PDP.github.io/), SIGGRAPH Asia 2024 | 공식 method 설명 | RL rollout의 noisy state와 corrective action을 활용한 직접 diffusion policy다. 배포 상태의 분포 이동을 고려한다는 교훈을 참고하되, 미래 모션 target을 action label과 혼동하지 않는다. |
| [MaskedMimic](https://arxiv.org/abs/2409.14393), 2024 | 저자 초록 | 부분적인 motion description을 완성하는 제어 문제. 제어 조건의 유연성을 참고한다. 이 논문을 diffusion 생성기 근거로 분류하지 않는다. |
| [SMP](https://yxmu.foo/smp-page/), SIGGRAPH 2026 | 공식 method와 응용 설명 | Diffusion을 고정된 모션 reward로 재사용한다. 목적이 온라인 모션 생성인 현재 프로젝트와 역할이 다르므로 기본 구성에 섞지 않는다. |
| [Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking](https://arxiv.org/abs/2604.17335), 2026-04 공개 | 저자 초록 | 지형 조건 생성과 RL tracking, 생성기 연결 후 tracker 보완 학습을 결합한다. Terrain을 단순한 평지 환경 변경으로 볼 수 없다는 판단에 참고한다. |

출판 상태는 확인한 자료 수준으로 기재했다. 초록·프로젝트 페이지 수준으로 확인한 연구에 대해 상세 하이퍼파라미터나 재현 비용을 추정 사실처럼 쓰지 않는다.

## 2. 이번 프로젝트에서 확정한 선택

**선택한 구조는 작은 목표 조건부 모션 생성기 + 고정된 reference tracker다.**

- 목표 위치·방향·시간·정지 조건은 구현·평가가 명확하고 자연어 모델을 요구하지 않는다.
- RL을 tracker 학습에 집중해, 만능 command teacher와 범용 tracker를 각각 만드는 부담을 줄인다.
- 목표는 물리 rollout의 endpoint로 재라벨링한다. 기존 공개 모션을 활용하되 실행 데이터와 원본 reference를 구분한다.
- 검색·MLP·Diffusion 비교와 실제 history/실행 길이 ablation으로 생성–실행 차이를 설명한다.
- 기여는 재현과 통제 실험이다. 논문 수준 novelty·실용적 우월성·데이터 부족 해결을 선결론으로 두지 않는다.

이는 논문들로부터 내린 설계 판단이며 해당 논문이 이 작은 데이터·하드웨어 조합의 성공을 보장한다는 뜻이 아니다.
추가 비용이 낮아진다는 가설도 tracker/data 준비 비용을 포함한 별도 실험 없이는 입증되지 않는다.

## 3. 데이터와 도구의 실제 출발점

[MimicKit DeepMimic 공식 안내](https://github.com/xbpeng/MimicKit/blob/main/docs/README_DeepMimic.md)는
dataset을 지정한 여러 reference 추종을 지원한다. 이 지원과 runtime에 생성 reference를 주입하는 완성된 API는 구분한다.
현재 로컬 설정·adapter로 연결 가능한지는 직접 검증해야 한다.

[DeepMimic 원본 amass 폴더](https://github.com/xbpeng/DeepMimic/tree/master/data/motions/amass)의
ready_to_walk / walk_to_ready는 출발·정지 후보다. 로컬 임시 복사본에서 파일 존재를 확인했지만,
내용 검증·형식 변환·사용 조건 확인·physics tracking은 아직 하지 않았다.
임시 경로를 재현 가능한 영구 데이터 경로로 사용하지 않는다.

## 4. 이번에 선택하지 않은 확장

- **상태·action 공동 diffusion:** 제어기 구조와 dataset 계약이 바뀐다. 현재 reference 생성과 함께 구현하지 않는다.
- **Diffusion reward / SMP:** 사람의 reward 설계 부담을 연구할 다른 방향이지만 현재 질문을 대체하므로 제외한다.
- **Terrain:** 지형 관측·발 착지·접촉·새 reference·tracker 보완이 필요하므로 후속 프로젝트로 둔다.
- **장애물:** 기본 보고서 후 평지 장애물 하나부터 검토한다. 외부 path planner의 기여와 모션 생성기의 기여를 분리한다.
- **Run:** 초기 필수 성과를 지연시키지 않도록 운동 범위 확장으로 남긴다.

이 선택은 새로운 아이디어를 계속 추가하기 위한 목록이 아니다.
추가 논문은 분석과 해석에 활용하고, 기본 연구 질문·구조·완료 기준은 유지한다.

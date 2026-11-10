"""Goal PPO+AMP entry point; shares runtime path resolution with command launcher."""
import sys
from pathlib import Path
from train_command_policy import main

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    for flag, value in [('--env-config', root/'configs/goal_teacher/stage_a.yaml'),
                        ('--agent-config', root/'configs/goal_teacher/agent.yaml'),
                        ('--out-dir', root/'outputs/goal_teacher_stage_a_seed0')]:
        if flag not in sys.argv:
            sys.argv.extend([flag, str(value)])
    main()

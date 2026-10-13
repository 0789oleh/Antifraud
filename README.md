# Neuro-Symbolic Fraud Detection

Fraud detection with differentiable rule induction and built-in explainability.

## Status
🚧 In development. Current milestones:
- [x] Data loading (IEEE-CIS, synthetic)
- [x] Temporal split with leakage prevention
- [x] Learnable discretizer
- [ ] Rule learner
- [ ] Training pipeline
- [ ] Serving API
- [ ] Streamlit demo

## Quickstart
\`\`\`bash

\`\`\`

## Architecture
\`\`\`
src/
├── data/         # loaders, splits, preprocessing, features
├── models/       # neural backbones (WIP)
├── rules/        # symbolic layer (WIP)
├── training/     # training loops (WIP)
└── inference/    # serving (WIP)
\`\`\`
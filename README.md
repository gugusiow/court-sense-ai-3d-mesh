# Court Sense AI — 3D Mesh Jump Shot Analysis

HKUST Final Year Project. This is an application that analyses individual basketball training footage, reconstructs a 3D mesh of the athlete’s jump shot, and compares it against a 3D mesh of a professional NBA player. From that comparison, the system aims to provide personalised tips for improving jump shot accuracy.

## Pipeline (work in progress)

1. **Video ingest** — upload or record training clips
2. **Pose / mesh reconstruction** — estimate body pose and build a 3D mesh from footage
3. **Temporal alignment (tbc)** — sync student vs professional shot phases (load → release → follow-through)
4. **Comparison metrics (tbc)** — elbow angle, release height/timing, torso lean, wrist flick, and related biomechanics
5. **Coaching layer** — map metric differences to personalised coaching tips
6. **App UI** — side-by-side 3D visualisation and tip summary

## Clone the repository

```bash
git clone https://github.com/gugusiow/court-sense-ai-3d-mesh.git
cd court-sense-ai-3d-mesh
```

## Commit message format

We follow a lightweight Conventional Commits style:

<type>(<scope>): <short subject>

Types: feat, fix, refactor, chore, docs, test, perf, style.

Examples:
- feat(pose): add 2D keypoint detector baseline
- fix(data): handle missing frames in video loader
- docs(readme): add instructions

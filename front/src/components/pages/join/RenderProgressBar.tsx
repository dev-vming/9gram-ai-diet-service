const RenderProgressBar = (curStep: number) => {
  const steps = 6;
  const progressBarSteps = [];

  for (let i = 1; i <= steps; i++) {
    const isActive = i <= curStep;
    progressBarSteps.push(
      <div key={i} className={`progress-step ${isActive ? 'active' : ''}`} />
    );
  }

  return progressBarSteps;
};

export default RenderProgressBar;

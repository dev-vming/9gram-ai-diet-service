interface chartProps {
  children: React.ReactNode;
}

export function NutritionBarChart({ children }: chartProps): JSX.Element {
  return (
    <svg
      viewBox='0 0 600 20'
      width='100%'
      height='10px'
      preserveAspectRatio='none'
    >
      {children}
    </svg>
  );
}

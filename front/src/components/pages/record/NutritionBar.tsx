import { BarProps } from './RecordTypes';

export const NutritionBar: React.FC<BarProps> = ({
  width,
  height,
  fill,
  className,
  style,
}) => {
  return (
    <>
      <rect
        rx='10'
        width={width}
        height={height}
        fill={fill}
        className={className}
        style={style}
      />
    </>
  );
};

export default NutritionBar;

import { useEffect, useState } from 'react';
import styles from '@components/pages/record/nutritiondonutchart.module.css';
import {
  MealDetailData,
  NutritionDonutChartProps,
  totalNutrientsType,
} from './RecordTypes';
import { nutrientNames, NutrientKey } from './recordMappingConstant';

const radius = 22;
const circumference = 2 * Math.PI * radius;

const initialSet = {
  carbohydrates: 0,
  proteins: 0,
  fats: 0,
  dietaryFiber: 0,
};

const NutritionDonutChart = ({
  totalNutrient,
  isShowingTotal,
  data,
  selectedMealNumber,
}: NutritionDonutChartProps) => {
  const [mealData, setMealData] = useState<MealDetailData>(data || {});
  const [animationTrigger, setAnimationTrigger] = useState(false);

  useEffect(() => {
    setMealData(data || {});
  }, [data]);

  useEffect(() => {
    const animationTimer = setTimeout(() => {
      setAnimationTrigger(true);
    }, 100);
    return () => {
      clearTimeout(animationTimer);
      setAnimationTrigger(false);
    };
  }, [isShowingTotal, selectedMealNumber]);

  let standard: totalNutrientsType | undefined;

  if (mealData && !mealData[selectedMealNumber]) {
    for (const key in mealData) {
      if (mealData[key].recommendNutrient) {
        standard = mealData[key].recommendNutrient;
        break;
      }
    }
  }

  const recommendNutrient =
    mealData[selectedMealNumber]?.recommendNutrient || standard || initialSet;

  const recommendNutrientValues = Object.values(recommendNutrient);

  const nutrientsData: totalNutrientsType | undefined = isShowingTotal
    ? mealData?.[selectedMealNumber]?.totalNutrient || initialSet
    : totalNutrient;

  const nutrients = Object.entries(nutrientsData as totalNutrientsType).map(
    ([key, value], idx) => {
      let nutrientKey = nutrientNames[key as NutrientKey];
      const nutrientRatio =
        value !== 0 && recommendNutrient
          ? value / recommendNutrientValues[idx]
          : 0;
      const strokeDashoffset =
        nutrientRatio < 1 ? circumference * (1 - nutrientRatio) : 0;

      const customStyle: React.CSSProperties & {
        '--initialOffset': string;
        '--finalOffset': string;
      } = {
        '--initialOffset': `${circumference}px`,
        '--finalOffset': `${strokeDashoffset}px`,
      };

      return {
        key: nutrientKey,
        value,
        nutrientRatio,
        strokeDashoffset,
        customStyle,
      };
    }
  );

  return (
    <div className={styles.nutrients}>
      {nutrients.map((nutrient, idx) => (
        <div key={idx} className={styles.nutrient}>
          <p className={styles.nutrientTitle}>{nutrient.key}</p>
          <div className={styles.progress_wrapper}>
            <svg
              className={styles.progress}
              width='48'
              height='48'
              viewBox='0 0 48 48'
            >
              <circle
                className={styles.frame}
                cx='24'
                cy='24'
                r='22'
                strokeWidth='3'
                strokeDashoffset='1'
                strokeDasharray={circumference}
              />
              <circle
                key={idx}
                className={`${nutrient.nutrientRatio <= 1 ? styles.bar : styles.overbar} ${animationTrigger ? styles.animatedBar : ''}`}
                cx='24'
                cy='24'
                r='22'
                strokeWidth='3'
                style={nutrient.customStyle}
                strokeDashoffset={nutrient.strokeDashoffset}
                strokeDasharray={circumference}
              />
            </svg>
            <p className={styles.percent}>
              {Math.floor(nutrient.nutrientRatio * 100)}%
            </p>
          </div>
          <p className={styles.gram}>
            {nutrient.value}/{recommendNutrientValues[idx]}g
          </p>
        </div>
      ))}
    </div>
  );
};

export default NutritionDonutChart;

import style from './mealnutritionanalysis.module.css';
import { NutritionBarChart } from './NutritionBarChart';
import NutritionBar from './NutritionBar';
import { useEffect, useState } from 'react';
import { MealNutritionAnalysisProps } from './RecordTypes';
import NutritionDonutChart from './NutritionDonutChart';
import MealGraphToggle from './MealGraphToggle';
import { mapSelectMealToMsg } from './recordMappingConstant';

const initialNutrients = {
  carbohydrates: 0,
  proteins: 0,
  fats: 0,
  dietaryFiber: 0,
};

const MealNutritionAnalysis = ({
  className,
  data,
  selectedMealNumber,
}: MealNutritionAnalysisProps) => {
  const [isShowingTotal, setIsShowingTotal] = useState(true);
  const [animationTrigger, setAnimationTrigger] = useState(false);

  const toggleShowingTitle = mapSelectMealToMsg[selectedMealNumber];
  const totalMealCalories = data?.[selectedMealNumber]?.totalCalories || 0;
  const totalCalories = data
    ? Object.values(data).reduce(
        (acc, meal) => acc + (meal?.totalCalories || 0),
        0
      )
    : 0;

  const calculateTotalNutrients = () => {
    if (!data) return initialNutrients;
    return Object.values(data).reduce(
      (acc, meal) => {
        if (meal?.totalNutrient) {
          acc.carbohydrates += meal?.totalNutrient.carbohydrates ?? 0;
          acc.proteins += meal.totalNutrient.proteins ?? 0;
          acc.fats += meal.totalNutrient.fats ?? 0;
          acc.dietaryFiber += meal.totalNutrient.dietaryFiber ?? 0;
        }
        return acc;
      },
      { ...initialNutrients }
    );
  };

  const drawGraphByNutrients = isShowingTotal
    ? data?.[selectedMealNumber]?.totalNutrient
    : calculateTotalNutrients();

  let goalCalories: number | undefined;

  if (
    data &&
    data[selectedMealNumber] &&
    data[selectedMealNumber].targetCalories
  ) {
    goalCalories = data[selectedMealNumber].targetCalories;
  } else if (data && !data[selectedMealNumber]) {
    for (const key in data) {
      if (data[key].targetCalories) {
        goalCalories = data[key].targetCalories;
        break;
      }
    }
  }

  const calculatePercentage = (calories: number) => {
    if (goalCalories !== undefined) {
      return calories === 0
        ? 0
        : Math.min(100, (calories / goalCalories) * 100);
    }
  };

  const percentage = isShowingTotal
    ? calculatePercentage(totalMealCalories)
    : calculatePercentage(totalCalories);

  const barFill = (percentage as number) >= 100 ? '#ff6a6a' : '#007bff';

  useEffect(() => {
    const animationTimer = setTimeout(() => {
      setAnimationTrigger(true);
    }, 100);
    return () => {
      clearTimeout(animationTimer);
      setAnimationTrigger(false);
    };
  }, [isShowingTotal, selectedMealNumber]);

  const handleSwitchGraph = () => {
    setIsShowingTotal(!isShowingTotal);
  };

  return (
    <>
      <div className={className}>
        <div className={style.headerContainer}>
          <div className={style.header}>
            <div className={style.title} onClick={handleSwitchGraph}>
              {isShowingTotal ? `${toggleShowingTitle} 섭취량` : '총 섭취량'}
            </div>
            <MealGraphToggle
              isShowingTotal={isShowingTotal}
              setIsShowingTotal={setIsShowingTotal}
            />
          </div>
          <div className={style.caloriesCount}>
            <div className={style.intakeCalories}>
              {' '}
              {isShowingTotal ? totalMealCalories : totalCalories || 0}{' '}
            </div>
            <div className={style.totalCalories}> /{goalCalories} </div>
          </div>
        </div>

        <NutritionBarChart>
          <NutritionBar
            key='goal-calories'
            width='100%'
            height='20px'
            fill='#edf3f9'
          />
          <NutritionBar
            key='consumed-calories'
            width='0%'
            className={`${animationTrigger ? style.startAnimation : ''}`}
            style={{ '--fillWidth': `${percentage}%` }}
            height='20px'
            fill={barFill}
          />
        </NutritionBarChart>

        <NutritionDonutChart
          totalNutrient={drawGraphByNutrients}
          isShowingTotal={isShowingTotal}
          data={data}
          selectedMealNumber={selectedMealNumber}
        />
      </div>
    </>
  );
};

export default MealNutritionAnalysis;

import style from './mealtag.module.css';
import { MealTagProps } from './RecordTypes';

const MealTag = ({ className, data, selectedMealNumber }: MealTagProps) => {
  const tagName = data?.[selectedMealNumber]?.foods || [];

  return (
    <div className={className}>
      <div className={style.containerName}> 먹은 음식 </div>
      {tagName.length > 0 ? (
        <div className={style.tagContainer}>
          {tagName.map((food, index) => (
            <span key={index} className={style.tagStyle}>
              {food.foodName}
            </span>
          ))}
        </div>
      ) : (
        <div className={style.guideComment}>오늘의 식단을 업로드 해주세요.</div>
      )}
    </div>
  );
};

export default MealTag;

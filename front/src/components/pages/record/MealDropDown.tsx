import { useState } from 'react';
import style from './mealpage.module.css';
import { mealTypes } from './recordMappingConstant';
export const MealDropDown = ({ selectedMeal, handleMealSelect }: any) => {
  const [isDropdownVisible, setDropdownVisible] = useState(false);

  return (
    <div className={style.mealToggle}>
      <div onClick={() => setDropdownVisible(!isDropdownVisible)}>
        {selectedMeal || '아침'} ▼
      </div>
      {isDropdownVisible && (
        <div className={style.dropdown}>
          {mealTypes.map((mealType) => (
            <div
              key={mealType}
              onClick={() => {
                handleMealSelect(mealType);
                setDropdownVisible(false);
              }}
              className={selectedMeal === mealType ? style.active : ''}
            >
              {mealType}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

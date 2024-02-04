export const mapSelectMealToMsg: { [key: string]: string } = {
  '1': '아침',
  '2': '점심',
  '3': '저녁',
  '4': '간식',
};

export const mealTypes = ['아침', '점심', '저녁', '간식'];

export const findMealNumber = (meal: string): string => {
  const mealNumber = Object.keys(mapSelectMealToMsg).find(
    (key) => mapSelectMealToMsg[key] === meal
  );
  return mealNumber || '1';
};

export type NutrientKey =
  | 'carbohydrates'
  | 'proteins'
  | 'fats'
  | 'dietaryFiber';
type NutrientNames = { [key in NutrientKey]: string };

export const nutrientNames: NutrientNames = {
  carbohydrates: '탄수화물',
  proteins: '단백질',
  fats: '지방',
  dietaryFiber: '식이섬유',
};

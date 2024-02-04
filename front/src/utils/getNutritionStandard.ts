import { UserData } from '@components/pages/my-page/MypageTypes';

const getNutritionStandard = (data: UserData) => {
  // const userGoal = data.dietGoal.toString();
  const userGoal = data.dietGoal;

  const userGoalCalories = data.targetCalories;
  const userGender = data.gender;
  const goalRatio: { [key: string]: number[] } = {
    '1': [3, 5, 2],
    '2': [4, 4, 2],
    '3': [5, 2, 3],
    '4': [4, 2, 4],
  };
  const [ratioX, ratioY, ratioZ] = goalRatio[userGoal.toString()];

  const k = userGender === '1' ? 38 : userGender === '2' ? 25 : 30;

  let z = 1;
  let x = (ratioX / ratioZ) * z;
  let y = (ratioY / ratioZ) * z;
  let totalCalories = 4 * x + 4 * y + 9 * z;

  while (totalCalories < userGoalCalories) {
    z++;
    x = (ratioX / ratioZ) * z;
    y = (ratioY / ratioZ) * z;
    totalCalories = 4 * x + 4 * y + 9 * z;
  }

  return totalCalories - userGoalCalories < 50
    ? {
        carbohydrates: Math.round(x),
        proteins: Math.round(y),
        fats: Math.round(z),
        dietaryFiber: k,
      }
    : { carbohydrates: 200, proteins: 100, fats: 40, dietaryFiber: 30 };
};

export default getNutritionStandard;

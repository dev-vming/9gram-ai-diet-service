export const recommendIntakeData = (targetCalories, recommendIntake) => {
  let recommendNutrient = {};
  if (recommendIntake === null) {
    recommendNutrient = {
      carbohydrates: 10,
      proteins: 10,
      fats: 10,
      dietaryFiber: 10,
    };
  } else {
    recommendNutrient = {
      carbohydrates: recommendIntake[0] === null ? 10 : recommendIntake[0],
      proteins: recommendIntake[1] === null ? 10 : recommendIntake[1],
      fats: recommendIntake[2] === null ? 10 : recommendIntake[2],
      dietaryFiber: recommendIntake[3] === null ? 10 : recommendIntake[3],
    };
  }
  return {
    targetCalories,
    recommendNutrient,
  };
};

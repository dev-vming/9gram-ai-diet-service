import { UserData } from './MypageTypes';

export const calBMR = ({ data }: { data: UserData }) => {
  const gender = Number(data.gender);
  const age = Number(data.age);
  const mapGenderToBmr: Record<number, number> = {
    1: 10 * data.weight + 6.25 * data.height - 5 * age + 5,
    2: 10 * data.weight + 6.25 * data.height - 5 * age - 161,
    3:
      data.height > 175
        ? 10 * data.weight + 6.25 * data.height - 5 * age + 5
        : 10 * data.weight + 6.25 * data.height - 5 * age - 161,
  };
  const bmr = mapGenderToBmr[gender];
  return bmr;
};

export const calBMRCalories = ({
  bmr,
  data,
}: {
  bmr: number;
  data: UserData;
}) => {
  const activityAmount = Number(data.activityAmount);

  const mapActivityToBmr: Record<number, number> = {
    1: bmr * 1.2,
    2: bmr * 1.3,
    3: bmr * 1.55,
    4: bmr * 1.7,
  };

  const activityBmr = mapActivityToBmr[activityAmount] || bmr;
  return activityBmr;
};

export const adjustCaloriesByGoal = ({
  data,
  bmrCalories,
}: {
  data: UserData;
  bmrCalories: number;
}) => {
  const dietGoal = Number(data.dietGoal);

  const mapAdjustingBmr: Record<number, number> = {
    1: bmrCalories + 400, //근육증량
    2: bmrCalories - 400, // 체중감량
    3: bmrCalories, //체중유지
    4: bmrCalories + 300,
  };

  const adjustingBmr = mapAdjustingBmr[dietGoal] || bmrCalories;
  return adjustingBmr;
};

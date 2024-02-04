export const userData = {
  email: 'elice@gmail.com',
  username: 'elice',
  password: 'Elice1234@!',
  birthday: '2011-04-26',
  gender: 2 as 1 | 2 | 3,
  age: 40,
  profileImage:
    'https://reclo.s3.ap-northeast-2.amazonaws.com/upload/itemImage/40962e2811b76d1ab5f661013f500061_%ED%9B%84%EB%93%9C.jpg',

  healthInfo: {
    weight: 90,
    height: 190,
    goal: 2 as 1 | 2 | 3 | 4,
    targetCalories: 2535,
    recommendIntake: {
      carbohydrates: 100,
      proteins: 60,
      fats: 30,
      dietaryFiber: 23,
    },
    activityAmount: 4 as 1 | 2 | 3 | 4,
  },
};

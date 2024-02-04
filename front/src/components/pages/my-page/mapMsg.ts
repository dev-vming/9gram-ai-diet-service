export const gendertoMsg: { [key: number]: string } = {
  1: '남성',
  2: '여성',
  3: '기타',
};

export const mapGoaltoMsg: { [key: number]: string } = {
  1: '근육증량',
  2: '체중감량',
  3: '체중유지',
  4: '체중증량',
};

export const mapActivitytoMsg: { [key: number]: string } = {
  1: '비활동적',
  2: '약간 활동적',
  3: '활동적',
  4: '매우 활동적',
};

export const mapNutrientstoMsg: { [key: string]: string } = {
  carbohydrates: '탄수화물',
  proteins: '단백질',
  fats: '지방',
  dietaryFiber: '식이섬유',
};

export const findKeyByValue = (
  msg: { [key: string]: string },
  value: string
): string | undefined => {
  for (let [key, val] of Object.entries(msg)) {
    if (val === value) {
      return key;
    }
  }
};

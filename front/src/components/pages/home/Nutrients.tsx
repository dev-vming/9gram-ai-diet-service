import styles from '@components/pages/home/nutrients.module.css';

interface Nutrient {
  carbohydrates: number;
  proteins: number;
  fats: number;
  dietaryFiber: number;
  [key: string]: number;
}

interface Props {
  totalNutrient: Nutrient;
  recommendNutrient: Nutrient;
}

const keyToKorean: Record<string, string> = {
  carbohydrates: '탄수화물',
  proteins: '단백질',
  fats: '지방',
  dietaryFiber: '식이섬유',
};

const radius = 22;
const circumference = 2 * Math.PI * radius;

const Nutrients = ({ totalNutrient, recommendNutrient }: Props) => {
  return (
    <div className={styles.nutrients}>
      {Object.keys(totalNutrient).map((idx) => (
        <div key={`nutrient-${idx}`} className={styles.nutrient}>
          <p className='b-small'>{keyToKorean[idx]}</p>
          <div className={styles.progress_wrapper}>
            <svg
              className={styles.progress}
              width='48'
              height='48'
              viewBox='0 0 48 48'
              overflow='visible'
            >
              <circle
                className={styles.frame}
                cx='24'
                cy='24'
                r='22'
                strokeWidth='3'
                strokeDashoffset='0'
                strokeDasharray={circumference}
              />
              <circle
                className={
                  totalNutrient[idx] / recommendNutrient[idx] <= 1
                    ? styles.bar
                    : styles.overbar
                }
                cx='24'
                cy='24'
                r='22'
                strokeWidth='3'
                strokeDashoffset={
                  totalNutrient[idx] / recommendNutrient[idx] < 1
                    ? circumference *
                      (1 - totalNutrient[idx] / recommendNutrient[idx])
                    : 0
                }
                strokeDasharray={circumference}
              />
            </svg>
            <p
              className={
                totalNutrient[idx] / recommendNutrient[idx] <= 1
                  ? `${styles.percent} b-tiny`
                  : `${styles.overpercent} b-tiny`
              }
            >
              {Math.round((totalNutrient[idx] / recommendNutrient[idx]) * 100)}%
            </p>
          </div>
          <p className={`${styles.gram} r-regular`}>
            {Math.round(totalNutrient[idx])}/{recommendNutrient[idx]}g
          </p>
        </div>
      ))}
    </div>
  );
};

export default Nutrients;

import { Plus } from '@assets/Plus';
import styles from '@components/pages/home/mealcard.module.css';
import { useNavigate } from 'react-router-dom';
const mealLogo = '/images/9gram_logo.png';

interface Props {
  date: string;
  dateArr: [number, number, string | undefined][];
}

interface Meal {
  [key: number]: string;
}

const mealtime: Meal = {
  1: '아침',
  2: '점심',
  3: '저녁',
  4: '간식',
};
const mealtimeList = ['아침', '점심', '저녁', '간식'];

const MealCard = ({ date, dateArr }: Props) => {
  const navigate = useNavigate();

  const handleClick = (idx: number) => {
    navigate(`/record/${date}/${idx + 1}`);
  };

  return (
    <div className={styles.mealcard_wrapper}>
      {mealtimeList.map((meal, idx) => (
        <div
          key={`mealcard-${idx}`}
          className={styles.mealcard}
          onClick={() => handleClick(idx)}
        >
          <>
            <div className={styles.block}></div>
            <p className={`${styles.nomeal} b-regular`}>{meal}</p>
            <Plus className={styles.noplus} width='25' />
            <p className={`${styles.nocalorie} b-medium`}>0kcal</p>
          </>
          {dateArr.length > 0 &&
            dateArr.map((date, idx2) =>
              mealtime[date[0]] === meal ? (
                <div key={`isMealcard-${idx2}`}>
                  <img
                    className={styles.img}
                    src={date[2] ? date[2] : mealLogo}
                    alt={mealtime[date[0]]}
                  />
                  <div className={styles.black}></div>
                  <p className={`${styles.meal} b-regular`}>
                    {mealtime[date[0]]}
                  </p>
                  <Plus className={styles.plus} width='25' />
                  <p className={`${styles.calorie} b-medium`}>{date[1]}kcal</p>
                </div>
              ) : null
            )}
        </div>
      ))}
    </div>
  );
};

export default MealCard;

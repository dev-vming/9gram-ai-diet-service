import { DownArrowLight } from '@assets/DownArrowLight';
import styles from '@components/pages/home/home.module.css';
import useCachingApi from '@hooks/useCachingApi';
import { useEffect, useState } from 'react';
import Calorie from './Calorie';
import MealCard from './MealCard';
import Nutrients from './Nutrients';

const week = ['일', '월', '화', '수', '목', '금', '토'];

const now = new Date();
const nowYear = now.getFullYear();
const nowMonth = now.getMonth();
const nowDate = now.getDate();
const nowDay = now.getDay();

const currentLastDayIndex = new Date(nowYear, nowMonth + 1, 0).getDate(); // 이번달 마지막 날짜
const currentFirstDayIndex = new Date(nowYear, nowMonth, 1).getDay(); // 이번달 첫번째 날
const totalWeek = Math.ceil(
  (currentLastDayIndex + currentFirstDayIndex + 1) / 7
); // 이번달 전체 주차
const nowWeek = Math.ceil((nowDate + 1) / 7); // 이번달 지금 주차

const Home = () => {
  const [selectedWeek, setSelectedWeek] = useState(nowWeek);
  const [selectedDay, setSelectedDay] = useState(nowDay);
  const [selectedNow, setSelectedNow] = useState(true);
  const [isOpen, setIsOpen] = useState(false);

  const todayDate = `${nowYear}-${nowMonth + 1 >= 10 ? nowMonth + 1 : `0${nowMonth + 1}`}-${nowDate >= 10 ? nowDate : `0${nowDate}`}`;
  const [date, setDate] = useState(todayDate);

  const [currentWeekArr, setCurrentWeekArr] = useState<number[]>([]);

  const { trigger, result }: { trigger: any; result: any } = useCachingApi({
    path: `/cumulative-record?date=${date}`,
    gcTime: 10000,
  });

  const triggerData = async () => {
    await trigger({});
  };

  useEffect(() => {
    triggerData();
  }, [date]);
  console.log(result);

  const handleClick = (idx: number) => {
    setSelectedDay(idx);
    setDate(
      `${nowYear}-${nowMonth + 1 >= 10 ? nowMonth + 1 : `0${nowMonth + 1}`}-${currentWeekArr[idx] >= 10 ? currentWeekArr[idx] : `0${currentWeekArr[idx]}`}`
    );
    if (idx === nowDay) {
      setSelectedNow(true);
    } else {
      setSelectedNow(false);
    }
  };

  const onToggle = () => setIsOpen(!isOpen);

  const onClickWeek = (week: number) => {
    setSelectedWeek(week);
  };

  const weekly = () => {
    const firstDate = (selectedWeek - 1) * 7 - currentFirstDayIndex + 1; // 해당주차 일요일 날짜
    const newWeekArr = [];
    for (let i = 0; i < 7; i++) {
      let date = firstDate + i;
      //마지막 데이터가 다음월이거나 처음 데이터가 전월인 경우
      if (date > currentLastDayIndex || date <= 0) {
        date = 0;
      }
      newWeekArr.push(date);
    }
    if (selectedWeek !== nowWeek || date !== todayDate) {
      for (let i = 0; i < 7; i++) {
        if (newWeekArr[i] !== 0) {
          setSelectedDay(i);
          setDate(
            `${nowYear}-${nowMonth + 1 >= 10 ? nowMonth + 1 : `0${nowMonth + 1}`}-${newWeekArr[i] >= 10 ? newWeekArr[i] : `0${newWeekArr[i]}`}`
          );
          break;
        }
      }
    }
    setCurrentWeekArr(newWeekArr);
    setIsOpen(false);
  };

  useEffect(() => {
    weekly();
  }, [selectedWeek]);

  return (
    <div className={styles.home_wrapper}>
      <div className={`${styles.select_wrapper} b-small`}>
        <p>
          {nowYear}.{String(nowMonth + 1).padStart(2, '0')}
        </p>
        <div onClick={onToggle} className={styles.dropdown_wrapper}>
          <div className={styles.dropdown_line}>
            {selectedWeek}주차 <DownArrowLight />
          </div>

          {isOpen && (
            <div className={styles.dropdown}>
              {new Array(totalWeek).fill(null).map((_, idx) => (
                <div
                  key={`week-${idx}`}
                  onClick={() => onClickWeek(idx + 1)}
                  className={idx + 1 === selectedWeek ? styles.active : ''}
                >
                  {idx + 1}주차
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className={styles.week}>
        {currentWeekArr.map((date, idx) => (
          <div className={styles.date_wrapper} key={idx}>
            {idx === 0 ? (
              <div className={`${styles.sunday} r-small`}>{week[idx]}</div>
            ) : idx === 6 ? (
              <div className={`${styles.saturday} r-small`}>{week[idx]}</div>
            ) : (
              <div className={`${styles.day} r-small`}>{week[idx]}</div>
            )}
            <div className={styles.date} onClick={() => handleClick(idx)}>
              {date === 0 ? (
                <></>
              ) : selectedDay === idx ? (
                <div className={`${styles.selected} r-large`}>{date}</div>
              ) : (
                <div className='r-large'>{date}</div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className={styles.today}>
        <p className={`${styles.title} b-regular`}>
          {selectedNow && currentWeekArr[selectedDay] === nowDate
            ? '오늘의'
            : `${week[selectedDay]}요일`}{' '}
          식단
        </p>
        {!result || result?.data.length === 0 ? (
          <>
            <Calorie totalCalories={0} recommendCalories={10} />
            <Nutrients
              totalNutrient={{
                carbohydrates: 0,
                proteins: 0,
                fats: 0,
                dietaryFiber: 0,
              }}
              recommendNutrient={{
                carbohydrates: 10,
                proteins: 10,
                fats: 10,
                dietaryFiber: 10,
              }}
            />
            <MealCard date={date} dateArr={[]} />
          </>
        ) : (
          <>
            <Calorie
              totalCalories={
                result?.data.totalCalories ? result?.data.totalCalories : 0
              }
              recommendCalories={result?.data.targetCalories}
            />
            <Nutrients
              totalNutrient={
                result?.data.totalNutrient
                  ? result?.data.totalNutrient
                  : {
                      carbohydrates: 0,
                      proteins: 0,
                      fats: 0,
                      dietaryFiber: 0,
                    }
              }
              recommendNutrient={result?.data.recommendNutrient}
            />
            <MealCard
              date={date}
              dateArr={
                (result?.data.dateArr as [number, number, string | undefined][])
                  ? result?.data.dateArr
                  : []
              }
            />
          </>
        )}
      </div>
    </div>
  );
};

export default Home;

import { useNavigate } from 'react-router-dom';
import MealImg from './MealImg';
import MealTag from './MealTag';
import MealNutritionAnalysis from './MealNutritionAnalysis';
import style from './mealdetail.module.css';
import { MealDetailProps } from './RecordTypes';

const MealDetail = ({ date, data, selectedMealNumber }: MealDetailProps) => {
  const navigate = useNavigate();

  return (
    <>
      <MealImg
        className={style.imgBox}
        date={date}
        data={data}
        selectedMealNumber={selectedMealNumber}
      />
      <div
        className={style.banner}
        onClick={() => {
          navigate(`/ai-analyze`);
        }}
      >
        <img
          className={style.bannerBackground}
          src='/images/recommendation_banner.png'
          alt='AI 식단 분석 서비스 이동 배너'
        />
        <div className={style.bannerFont}>
          {' '}
          AI가 말아주는 오늘의 한 끼 추천이 궁금하다면?{' '}
        </div>
      </div>
      <MealTag
        data={data}
        selectedMealNumber={selectedMealNumber}
        className={style.mealTag}
      />
      <MealNutritionAnalysis
        data={data}
        selectedMealNumber={selectedMealNumber}
        className={style.nutritionBox}
      />
    </>
  );
};

export default MealDetail;

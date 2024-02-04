import { useNavigate } from 'react-router-dom';
import style from './mealImg.module.css';
import ImgTagContent from './ImgTagContent';
import { MealImgProps } from './RecordTypes';
import getDates from '@utils/getDates';

const MealImg = ({
  className,
  date,
  data,
  selectedMealNumber,
}: MealImgProps) => {
  const navigate = useNavigate();
  const { thisYear, thisMonth, thisDay } = getDates();
  const todayDate = `${thisYear}-${thisMonth}-${thisDay}`;

  const imgUrl =
    data && data[selectedMealNumber]
      ? data[selectedMealNumber].imgUrl
      : undefined;
  const foods =
    data && data[selectedMealNumber] ? data[selectedMealNumber].foods : [];

  return (
    <>
      <div className={className}>
        {imgUrl || foods?.length > 0 ? (
          <ImgTagContent
            data={data}
            selectedMealNumber={selectedMealNumber}
            imgUrl={imgUrl || '/images/ggu_logo_record.png'}
            className={className}
          />
        ) : (
          <div
            onClick={() => {
              navigate(`/add-photo/${date}/${selectedMealNumber}`);
            }}
          >
            {todayDate >= date && (
              <>
                <div className={style.uploadButton}> + </div>
                <div className={style.uploadText}>
                  {' '}
                  오늘의 식단을 <br /> 추가해주세요{' '}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );
};

export default MealImg;

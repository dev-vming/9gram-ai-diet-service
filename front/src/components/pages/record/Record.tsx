import { useNavigate, useParams } from 'react-router-dom';
import style from './record.module.css';
import getDates from '@utils/getDates';
import { useState, useEffect } from 'react';
import { mapSelectMealToMsg, mealTypes } from './recordMappingConstant';
import useApi, { TriggerType } from '@hooks/useApi';
import { Modal, mapSelectModalMsg } from '@components/UI/Modal';
import { RecordProps } from './RecordTypes';
import { Plus } from '@assets/Plus';
const mealLogo = '/images/9gram_logo_box.png';

const Record = () => {
  const params = useParams();
  const selectedDate = params.selectedDate;
  const { thisYear, thisMonth, thisDay } = getDates();
  const todayDate = `${thisYear}-${thisMonth}-${thisDay}`;
  const dateSplit = selectedDate
    ? selectedDate.split('-')
    : todayDate.split('-');

  const [foodData, setFoodData] = useState<RecordProps>({
    dateArr: [[1], [2], [3], [4]] as unknown as Array<
      [number, number, string | null]
    >,
  });
  const [showModal, setShowModal] = useState(false);
  const [modalSelect, setModalSelect] = useState('');
  const [modalMsg, setModalMsg] = useState('');
  const [selectedMealDelete, setSelectedMealDelete] = useState<null | number>(
    null
  );

  const {
    trigger,
    result: data,
  }: {
    trigger: TriggerType;
    result: { data: RecordProps } | undefined;
  } = useApi({
    path: `/cumulative-record/meal?date=${dateSplit[0]}-${dateSplit[1]}-${dateSplit[2]}`,
  });

  useEffect(() => {
    trigger({});
  }, [data?.data?.dateArr?.length]);

  useEffect(() => {
    if (data && data.data.dateArr?.length > 0) {
      setFoodData(data.data);
    }
  }, [data]);

  const formatDate =
    dateSplit.length === 3
      ? `${dateSplit[0]}.${dateSplit[1]}.${dateSplit[2]}`
      : `${thisYear}.${thisMonth}.${thisDay}`;

  const headerDate = formatDate || `${thisYear}.${thisMonth}.${thisDay}`;

  const navigate = useNavigate();

  const handleMealClick = (meal: number) => {
    navigate(`/record/${selectedDate || todayDate}/${meal}`);
  };

  const handleShowDelteModal = (
    meal: number,
    e: React.MouseEvent<HTMLImageElement>
  ) => {
    e.stopPropagation();
    setShowModal(true);
    setModalSelect('mealDelete');
    setModalMsg(mapSelectModalMsg.mealDelete);
    setSelectedMealDelete(meal);
  };

  const handleMealDelete = () => {
    if (selectedMealDelete === null) return;
    const updatedFoodData = { ...foodData };
    updatedFoodData.dateArr[selectedMealDelete - 1] = [
      selectedMealDelete,
      0,
      null,
    ];
    trigger({
      method: 'delete',
      path: `/records?date=${todayDate}&mealType=${selectedMealDelete}`,
    });
    setFoodData(updatedFoodData);
    setShowModal(false);
    setSelectedMealDelete(null);
  };

  const handleConfirm = () => {
    if (modalSelect === 'mealDelete' && selectedMealDelete !== null) {
      handleMealDelete();
    }
  };

  return (
    <div>
      {showModal && (
        <Modal
          modalSelect={modalSelect}
          modalMsg={modalMsg}
          onClose={() => setShowModal(false)}
          onConfirm={handleConfirm}
        />
      )}
      <div className={style.meal_container}>
        <div className={style.meal_header}> {headerDate} </div>
        {foodData &&
          foodData.dateArr.map((mealData, index) => (
            <div
              onClick={() => handleMealClick(mealData[0])}
              key={index}
              className={style.meal_content}
            >
              <div className={style.meal_info}>
                {mealData[1] || mealData[2] ? (
                  <>
                    <img
                      className={style.meal_contentBackground}
                      src={mealData[2] || mealLogo}
                      alt='하루 식단 이미지'
                    />

                    <div className={style.meal_time}>
                      {mapSelectMealToMsg[mealData[0]]}
                    </div>
                    <div className={style.meal_calories}>
                      {' '}
                      {Math.round(mealData[1]) ?? 0} kcal{' '}
                    </div>
                  </>
                ) : (
                  <>
                    <div className={style.defaultBackground}></div>
                    <div className={style.default_time}>
                      {mapSelectMealToMsg[mealData[0]]}
                    </div>
                    <div className={style.default_calories}> 0 kcal </div>
                  </>
                )}
              </div>
              {!mealData[1] && !mealData[2] ? (
                <Plus
                  className={style.meal_plusButton}
                  onClick={(e) => handleMealClick(mealData[0])}
                />
              ) : (
                <img
                  className={style.meal_button}
                  src={'/icons/meal_delete.png'}
                  onClick={(e) => handleShowDelteModal(mealData[0], e)}
                  alt={'하루 식단 삭제 버튼'}
                />
              )}
            </div>
          ))}
      </div>
    </div>
  );
};

export default Record;

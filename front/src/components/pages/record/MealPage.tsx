import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import style from './mealpage.module.css';
import ButtonCommon from '@components/UI/ButtonCommon';
import getDates from '@utils/getDates';
import MealDetail from './MealDetail';
import { MealDropDown } from './MealDropDown';
import { MealDetailData } from './RecordTypes';
import useApi from '@hooks/useApi';
import { Modal, mapSelectModalMsg } from '@components/UI/Modal';
import { mapSelectMealToMsg } from './recordMappingConstant';
import { findMealNumber } from './recordMappingConstant';

const MealPage = () => {
  const params = useParams();
  const navigate = useNavigate();
  const { thisYear, thisMonth, thisDay } = getDates();
  const todayDate = `${thisYear}-${thisMonth}-${thisDay}`;
  const date = params.date || todayDate;
  const selectedMealTime = params.mealTime || '1';
  const dateSplit = date.split('-');
  const formatDate = `${dateSplit[0]}.${dateSplit[1]}.${dateSplit[2]}`;

  const mealMsg = mapSelectMealToMsg[selectedMealTime];
  const [selectedMeal, setSelectedMeal] = useState(mealMsg);
  const [selectedMealNumber, setSelectedMealNumber] = useState(
    Number(findMealNumber(selectedMeal))
  );
  const [data, setData] = useState<MealDetailData>({});
  const [imgData, setImgData] = useState('');
  const [coordinate, setCoordinate] = useState({});
  const [isDropdownVisible, setDropdownVisible] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalSelect, setModalSelect] = useState('');
  const [modalMsg, setModalMsg] = useState('');

  type mealDataType = {
    data: MealDetailData;
  };
  const { trigger, result, error, loading } = useApi<mealDataType>({});

  useEffect(() => {
    trigger({
      path: `/records?date=${date}`,
    });
  }, [date]);

  useEffect(() => {
    if (!result?.data) return;
    const mealData = result.data;
    if (mealData) {
      setData(result.data);
      if (mealData && mealData[selectedMealNumber]) {
        setCoordinate(mealData[selectedMealNumber].foods);
        setImgData(mealData[selectedMealNumber].imgUrl || '');
      }
    } else {
      setData({});
      setCoordinate([]);
      setImgData('');
    }
  }, [result?.data]);

  useEffect(() => {
    if (selectedMealNumber) {
      navigate(`/record/${date}/${selectedMealNumber}`);
    }
  }, [selectedMealNumber, result?.data]);

  const handleMealSelect = (mealType: string) => {
    setSelectedMeal(mealType);
    const newMealNumber = Number(findMealNumber(mealType));
    setSelectedMealNumber(newMealNumber);
    setDropdownVisible(false);
  };

  const handleShowDelteModal = () => {
    setShowModal(true);
    setModalSelect('mealDelete');
    setModalMsg(mapSelectModalMsg.mealDelete);
  };

  const handleMealDelete = () => {
    trigger({
      method: 'delete',
      path: `/records?date=${date}&mealType=${selectedMealNumber}`,
    });
    const deleteData = result?.data;
    const updatedData = { ...deleteData };
    delete updatedData[selectedMealNumber];
    setData(updatedData);
    navigate(`/record/${date}`);
  };

  const handleConfirm = () => {
    if (modalSelect === 'mealDelete') {
      handleMealDelete();
    }
  };

  return (
    <>
      {showModal && (
        <Modal
          modalSelect={modalSelect}
          modalMsg={modalMsg}
          onClose={() => setShowModal(false)}
          onConfirm={handleConfirm}
        />
      )}
      <div className={style.container}>
        <div className={style.pageTitle}>
          <div>{formatDate}</div>
          <MealDropDown
            selectedMeal={selectedMeal}
            handleMealSelect={handleMealSelect}
          />
          <div className={style.headerButton}>
            {data[selectedMealNumber] &&
              data[selectedMealNumber].foods?.length > 0 && (
                <>
                  <ButtonCommon
                    variant='default-active'
                    size='tiny'
                    onClick={() => {
                      navigate(`/record/${date}/${selectedMealTime}/edit`, {
                        state: { foods: coordinate, imgUrl: imgData },
                      });
                    }}
                  >
                    <> 수정 </>
                  </ButtonCommon>
                  <ButtonCommon
                    variant='default-active'
                    size='tiny'
                    onClick={handleShowDelteModal}
                  >
                    <> 삭제 </>
                  </ButtonCommon>
                </>
              )}
          </div>
        </div>
        <MealDetail
          date={date}
          data={data}
          selectedMealNumber={selectedMealNumber}
        />
      </div>
    </>
  );
};

export default MealPage;

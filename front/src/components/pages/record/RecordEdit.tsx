import { useLocation, useNavigate, useParams } from 'react-router-dom';
import styles from './recordedit.module.css';
import RecordEditDetail from './RecordEditDetail';
import { useEffect, useMemo, useRef, useState } from 'react';
import ButtonCommon from '@components/UI/ButtonCommon';
import { MergingTags } from './MergingTags';
import useApi from '@hooks/useApi';

interface Food {
  foodName: string;
  XYCoordinate: number[];
  counts: number;
  foodInfoId?: string;
  calories?: number;
  carbohydrates?: number;
  dietaryFiber?: number;
  fats?: number;
  proteins?: number;
  totalCapacity?: number;
  recordId?: string;
}

interface MealTime {
  [key: string]: string;
}

interface Result {
  config: {};
  data: string;
  headers: {};
  request: {};
  status: number;
  statusText: string;
}

interface FoodInfoList {
  foodInfoIdList: string[];
}

interface SearchIdResult {
  config: {};
  data: FoodInfoList;
  headers: {};
  request: {};
  status: number;
  statusText: string;
}

const RecordEdit = () => {
  const navigate = useNavigate();
  const { state } = useLocation();

  const params = useParams();
  const date = params.date;
  const mealTime = params.mealTime;
  const dateSplit = date?.split('-');
  const mealTimetoStr: MealTime = {
    '1': '아침',
    '2': '점심',
    '3': '저녁',
    '4': '간식',
  };

  const [foods, setFoods] = useState<Food[]>([
    {
      foodName: '',
      XYCoordinate: [],
      counts: 1,
    },
  ]);
  const [imgUrl, setImgUrl] = useState('');

  function base64toFile(base_data: string, filename: string) {
    if (foods && foods[foods.length - 1]?.hasOwnProperty('recordId')) return;
    const arr = base_data.split(',');
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);

    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }

    return new File([u8arr], filename, { type: 'image/jpg' });
  }

  const fileName = useMemo(() => {
    if (foods && foods[foods.length-1]?.hasOwnProperty('recordId')) return;
    return (
      date + ' ' + mealTimetoStr[mealTime as string] + Math.random() + '.jpg'
    );
  }, []);
  const file = 
    foods  && !(foods[foods.length-1]?.hasOwnProperty('recordId')) && imgUrl && fileName && base64toFile(imgUrl, fileName);

  useEffect(() => {
    if (state) {
      setFoods(state.foods);
      setImgUrl(state.imgUrl);
    }
  }, []);

  const [focus, setFocus] = useState<number | undefined>();

  const handleFocus = (
    e: React.MouseEvent<HTMLCanvasElement | HTMLDivElement, MouseEvent>
  ) => {
    setFocus(foods.findIndex((food) => food.foodName === e.currentTarget.id));
  };

  const addFood = (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    foods[0] && foods[0].foodName === '음식명'
      ? alert('음식 상세 정보를 추가해주세요!')
      : setFoods([
          {
            foodName: '음식명',
            XYCoordinate: [],
            counts: 1,
            foodInfoId: '',
          },
          ...foods,
        ]);
    e.currentTarget.nextElementSibling?.scrollTo({
      left: 0,
      behavior: 'smooth',
    });
  };

  const deletefood = (e: React.MouseEvent<HTMLImageElement, MouseEvent>) => {
    if (foods.length > 1) {
      const delete_target = e.currentTarget.id;
      setFoods(foods.filter((food) => food.foodName !== delete_target));
    } else {
      alert('음식은 1개 이상 등록되어야 합니다!');
    }
  };

  const scrollDiv = useRef<HTMLDivElement | null>(null);
  const scrollConvert = (e: React.WheelEvent<HTMLDivElement>) => {
    const { deltaY } = e;
    const el: any = scrollDiv.current;

    el.scrollTo({
      left: el.scrollLeft + deltaY,
      behavior: 'smooth',
    });
  };

  const handleEnter = () => {
    const main = document.querySelector('.main') as HTMLElement;
    main.style.overflow = 'hidden';
  };

  const handleLeave = () => {
    const main = document.querySelector('.main') as HTMLElement;
    main.style.overflow = 'scroll';
  };

  const canvasRef = useRef<null[] | HTMLCanvasElement[]>([]);

  const createCanvas = (food: Food, index: number) => {
    const canvas = canvasRef.current[index];
    const context = canvas?.getContext('2d');

    const image = new Image();
    image.src = imgUrl;

    const cropX = food.XYCoordinate[0] * image.width - 45;
    const cropY = food.XYCoordinate[1] * image.height - 45;

    image.onload = () => {
      context?.drawImage(image, cropX, cropY, 90, 90, 0, 0, 90, 90);
    };
  };

  useEffect(() => {
    foods.map((food, index) => {
      return createCanvas(food, index);
    });
  }, [foods]);

  const presignedUrl = useApi<Result>({
    method: 'post',
  });

  const s3Upload = useApi<Result>({
    method: 'put',
  });

  const recordPost = useApi<Result>({
    method: 'post',
  });

  const recordPut = useApi<Result>({
    method: 'put',
  });

  const getFoodInfoId = useApi<SearchIdResult>({
    method: 'post',
  });

  useEffect(() => {
    if (!foods) return;
    if (foods.length === 0) return;
    if (foods[0].foodName === '음식명') return;
    if (foods[0].foodName === '') return;
    if (foods[0].foodInfoId) return;
    const foodList =
      foods.length === 1
        ? [foods[0].foodName]
        : foods.map((food) => food.foodName);
    getFoodInfoId.trigger({
      path: '/food-info/foods',
      data: { foodList },
    });
  }, [foods]);

  useEffect(() => {
    if (!getFoodInfoId.result) return;
    const foodInfoIdList = getFoodInfoId.result.data.foodInfoIdList;
    const newFoods = foods.length === 1 ? foods[0] : [...foods];
    foods.length === 1
      ? (foods[0].foodInfoId = foodInfoIdList[0])
      : (newFoods as Food[]).map((food, index) => {
          food.foodInfoId = foodInfoIdList[index];
        });
    foods.length === 1
      ? setFoods([newFoods] as Food[])
      : setFoods(newFoods as Food[]);
  }, [getFoodInfoId.result]);

  useEffect(() => {
    if (imgUrl === undefined || imgUrl === '') return;
    if (presignedUrl.result && !file) return;
    if (foods[0] && foods[0].recordId) return;
    presignedUrl.trigger({
      path: `/image/presigned-url/food/${fileName}`,
      data: { fileName },
    });
  }, [imgUrl]);

  useEffect(() => {
    if (imgUrl === undefined) return;
    if (foods[0] && foods[0].recordId) return;
    if (!presignedUrl.result && !file) return;
    s3Upload.trigger({
      path: presignedUrl.result?.data,
      data: file,
    });
  }, [presignedUrl.result?.data]);

  const editDone = () => {
    const pUrl = imgUrl && (foods && !foods[foods.length-1].recordId) && presignedUrl.result?.data.split('?')[0];
    if (foods.length === 0) return;
    if (!foods || foods[0].foodName === '' || !foods[0].foodInfoId) return;
    const newFoods = [...foods];
    newFoods.map((food: Food) => {
      delete food.calories;
      delete food.carbohydrates;
      delete food.dietaryFiber;
      delete food.fats;
      delete food.proteins;
      delete food.totalCapacity;
    });

    if (newFoods[newFoods.length-1].recordId) {
      recordPut.trigger({
        path: `/records?date=${date}&mealType=${mealTime}`,
        data: {
          mealType: mealTime,
          imgUrl: imgUrl,
          foods: newFoods,
        },
      });
    }

    if (!newFoods[newFoods.length-1].recordId) {
      recordPost.trigger({
        path: `/records?date=${date}&mealType=${mealTime}`,
        data: {
          mealType: mealTime,
          imgUrl: pUrl,
          foods: newFoods,
        },
      });
    }
  };

  useEffect(() => {
    if (!recordPost.result) return;
    if (recordPost.result.data === '식단 기록 성공') {
      navigate(`/record/${date}/${mealTime}`);
    }
  }, [recordPost.result]);

  useEffect(() => {
    if (!recordPut.result) return;
    if (recordPut.result.data === '식단 수정 성공') {
      navigate(`/record/${date}/${mealTime}`);
    }
  }, [recordPut.result]);

  return (
    <>
      <div className={styles.datebox}>
        {dateSplit && mealTime && (
          <p className='b-small'>
            {dateSplit[0]}. {dateSplit[1]}. {dateSplit[2]}{' '}
            {mealTimetoStr[mealTime]}
          </p>
        )}
      </div>

      <div className={styles.imgbox} style={{ position: 'relative' }}>
        <MergingTags tagData={foods} />
        <img
          className={styles.mealimg}
          src={imgUrl || '/images/9gram_logo.png'}
          alt='식단이미지'
        />
      </div>

      <div className={styles.photobox}>
        <button className={styles.addphotobtn} onClick={(e) => addFood(e)}>
          <img
            className={styles.btnicon}
            src='/icons/plusicon.png'
            alt='음식추가버튼'
          />
          <p className={styles.btntext}>
            음식
            <br />
            추가
          </p>
        </button>
        <div
          className={styles.tagbox}
          ref={scrollDiv}
          onWheel={(e) => scrollConvert(e)}
          onMouseEnter={handleEnter}
          onMouseLeave={handleLeave}
        >
          {foods &&
            foods.map((food: Food, index: number) => (
              <div key={index} className={styles.tagitem}>
                <div className={styles.tagimgwrap}>
                  {food.XYCoordinate.length === 0 ||
                  food.XYCoordinate[0] === 0? (
                    <img
                      className={`${styles.tagimg} ${
                        focus === index && styles.focusimg
                      }`}
                      id={food.foodName}
                      src='/images/9gram_logo.png'
                      alt={food.foodName}
                      onClick={(e) => handleFocus(e)}
                    />
                  ) : (
                    <canvas
                      className={`${styles.tagimg} ${
                        focus === index && styles.focusimg
                      }`}
                      id={food.foodName}
                      ref={(element) => {
                        canvasRef.current[index] = element;
                      }}
                      width={90}
                      height={90}
                      onClick={(e) => handleFocus(e)}
                    />
                  )}

                  <img
                    className={styles.tagdeleteicon}
                    id={food.foodName}
                    src='/icons/deleteicon.png'
                    alt='태그삭제'
                    onClick={(e) => deletefood(e)}
                  />
                </div>
                <div style={{ width: '90px' }}>
                  <p
                    className={`${
                      focus === index ? styles.focustxt : styles.tagtxt
                    }`}
                  >
                    {food.foodName}
                  </p>
                </div>
              </div>
            ))}
        </div>
      </div>
      {focus !== undefined && (
        <RecordEditDetail focus={focus} foods={foods} setFoods={setFoods} />
      )}

      <div className={styles.btnbox}>
        {focus === undefined ? (
          <ButtonCommon
            size='medium'
            variant='disabled'
            onClick={() => navigate(-1)}
          >
            취소
          </ButtonCommon>
        ) : (
          <ButtonCommon
            size='medium'
            variant='disabled'
            onClick={() => setFocus(undefined)}
          >
            취소
          </ButtonCommon>
        )}

        {foods && foods.length && foods[0].foodName !== '음식명' ? (
          <ButtonCommon
            size='medium'
            variant='default-active'
            onClickBtn={editDone}
          >
            수정 완료
          </ButtonCommon>
        ) : (
          <ButtonCommon size='medium' variant='default' disabled={true}>
            수정 완료
          </ButtonCommon>
        )}
      </div>
    </>
  );
};

export default RecordEdit;

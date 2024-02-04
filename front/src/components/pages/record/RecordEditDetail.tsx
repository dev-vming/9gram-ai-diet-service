import InputCommon from '@components/UI/InputCommon';
import styles from './recordeditdetail.module.css';
import ButtonCommon from '@components/UI/ButtonCommon';
import { useEffect, useMemo, useState } from 'react';
import useApi from '@hooks/useApi';
import useIntersect from '@hooks/useIntersect';
import getFoodId from '@utils/getFoodId';

interface Nutrition {
  name: string;
  gram: number;
}
interface FoodInfo {
  foodInfoId: string;
  foodName: string;
}

interface Food {
  foodName: string;
  XYCoordinate: number[];
  recordId?: string;
  counts: number;
  foodInfoId?: string;
  calories?: number;
  carbohydrates?: number;
  dietaryFiber?: number;
  fats?: number;
  proteins?: number;
  totalCapacity?: number;
}

interface Props {
  focus: number | undefined;
  foods: Food[];
  setFoods: React.Dispatch<React.SetStateAction<Food[]>>;
}

interface Result {
  data: Food;
}

interface SearchResult {
  data: FoodInfo[];
}


const RecordEditDetail = ({ focus, foods, setFoods }: Props) => {
  const [searchInput, setSearchInput] = useState('');
  const [searching, setSearching] = useState(false);

  const focusing = foods[focus as number];

  const byFoodName = `foodName=${focusing.foodName}`;
  const byFoodInfoId = `foodInfoId=${focusing.foodInfoId}`;
  const path = focusing.foodInfoId
    ? `/food-info/foods?${byFoodInfoId}`
    : `/food-info/foods?${byFoodName}`;

  const { result, trigger } = useApi<Result>({
    path: path,
  });

  useEffect(() => {
    if (focus === undefined) return;
    if (focusing.calories) return;
    if (focusing.foodName === "음식명") return;
    trigger({});
  }, [foods[focus as number]]);

  useEffect(() => {
    if (!result) return;
    if (focus === undefined) return;
    let res = result.data;
    res.foodName = focusing.foodName;
    res.counts = focusing.counts;
    res.XYCoordinate = focusing.XYCoordinate;
    focusing.recordId && (res.recordId = focusing.recordId);
    const newFocusFood = { ...res };
    let copyFoods = [...foods];
    copyFoods[focus] = newFocusFood;
    setFoods(copyFoods);
  }, [result]);

  const [searchResults, setSearchResults] = useState(['']);
  const [foodInfo, setFoodInfo] = useState<FoodInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const firstSearch = useApi<SearchResult>({
    path: `/food-info/foods?keyword=${searchInput}`,
  });

  const handleSearch = () => {
    if (!searchInput) return;
    if (searchResults.length >= 10) return;
    firstSearch.trigger({});
  };

  useEffect(() => {
    if (!firstSearch.result) return;
    const newResults = firstSearch.result.data.map((food: FoodInfo) =>
      food.foodName.includes('_') ? food.foodName.split('_')[1] : food.foodName
    );
    setSearchResults(newResults);
    setFoodInfo(firstSearch.result.data);
    setSearching(true);
  }, [firstSearch.result]);

  const lastFoodId =
    foodInfo.length > 0 && foodInfo[foodInfo.length - 1].foodInfoId;

  const scrollSearch = useApi<SearchResult>({
    path: `/food-info/foods?keyword=${searchInput}&lastFoodId=${lastFoodId}`,
  });

  const onIntersect: IntersectionObserverCallback = async (
    [entry],
    observer
  ) => {
    if (entry.isIntersecting) {
      if (scrollSearch.result?.data?.length === 0) return;
      setIsLoading(true);
      observer.unobserve(entry.target);
      await scrollSearch.trigger({});
      observer.observe(entry.target);
      setIsLoading(false);
    }
  };

  const { setTarget } = useIntersect({
    root: null,
    rootMargin: '0px',
    threshold: 0.5,
    onIntersect,
  });

  useEffect(() => {
    if (!scrollSearch.result) return;
    const newResults = scrollSearch.result.data.map((food: FoodInfo) =>
      food.foodName.includes('_') ? food.foodName.split('_')[1] : food.foodName
    );
    setSearchResults(searchResults.concat(newResults));
    setFoodInfo(foodInfo.concat(scrollSearch.result.data));
  }, [scrollSearch.result]);

  const editName = (e: React.MouseEvent<HTMLParagraphElement, MouseEvent>) => {
    if (foods.some((food) => food.foodName === e.currentTarget.textContent)) {
      alert('이미 등록된 음식입니다!');
    } else {
      let newArr = [...foods];
      if (focus === undefined) return;

      const newObj = newArr[focus].recordId ? {
        foodName: e.currentTarget.textContent as string,
        XYCoordinate: newArr[focus].XYCoordinate,
        counts: newArr[focus].counts,
        foodInfoId: getFoodId(
          e.currentTarget.textContent as string,
          foodInfo,
          searchInput
        ) as string,
        recordId: newArr[focus].recordId
      }: {
        foodName: e.currentTarget.textContent as string,
        XYCoordinate: newArr[focus].XYCoordinate,
        counts: newArr[focus].counts,
        foodInfoId: getFoodId(
          e.currentTarget.textContent as string,
          foodInfo,
          searchInput
        ) as string,
      };
      newArr[focus] = newObj;
      setFoods(newArr);
      setSearching(false);
    }
  };

  const increment = () => {
    if (focus === undefined) return;
    let copyArr = [...foods];
    copyArr[focus].counts = Number(copyArr[focus].counts) + 0.25;
    setFoods(copyArr);
  };

  const decrement = () => {
    if (focus === undefined) return;
    if (focusing.counts === 0.25) {
      alert('더 이상 양을 줄일 수 없습니다!');
    } else {
      let copyArr = [...foods];
      copyArr[focus].counts = Number(copyArr[focus].counts) - 0.25;
      setFoods(copyArr);
    }
  };

  const [calCH, calPT, calFT, calDF] = useMemo(
    () =>
      [
        focusing.carbohydrates,
        focusing.proteins,
        focusing.fats,
        focusing.dietaryFiber,
      ].map((data) => data && Math.round(data * focusing.counts)),
    [focusing]
  );

  const nutrients = [
    { name: '탄수화물', gram: calCH ? calCH : 0 },
    { name: '단백질', gram: calPT ? calPT : 0 },
    { name: '지방', gram: calFT ? calFT : 0 },
    { name: '식이섬유', gram: calDF ? calDF : 0 },
  ];

  return (
    <div className={styles.container}>
      <div className={styles.titlebox}>
        <p className='s-large'>{focusing.foodName}</p>
        <p className='r-super' style={{ marginLeft: 'auto' }}>
          {focusing.calories
            ? Math.round(focusing.counts * focusing.calories)
            : 0}
          Kcal
        </p>
      </div>

      <div className={styles.nutrientbox}>
        {result &&
          nutrients.map((nutrition: Nutrition, index) => (
            <div key={index} className={styles.circle}>
              <p className='s-regular'>{nutrition.name}</p>
              <p className='b-small' style={{ color: 'var(--main-blue)' }}>
                {nutrition.gram}g
              </p>
            </div>
          ))}
      </div>

      <div className={styles.searchbox}>
        <p
          className='r-medium'
          style={{
            color: 'var(--main-skyblue)',
            textAlign: 'left',
            marginBottom: '9px',
          }}
        >
          음식을 수정하려면 검색하세요!
        </p>
        <div className={styles.searchform}>
          <InputCommon
            size='medium'
            variant='default'
            value={searchInput}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              setSearchInput(e.target.value);
              setSearching(false);
              setSearchResults([]);
              setFoodInfo([]);
            }}
          />
          <ButtonCommon
            size='small'
            variant='default-active'
            onClickBtn={handleSearch}
          >
            검색
          </ButtonCommon>
          {searching && (
            <div className={styles.resultbox}>
              <div style={{ overflowY: 'auto' }}>
                {searchInput && searchResults.length > 0 ? (
                  searchResults.map((result: string, index) => (
                    <div key={index}>
                      <p
                        className='r-medium'
                        style={{ marginBottom: '5px' }}
                        onClick={(e) => editName(e)}
                      >
                        {result.split(searchInput)[0] !== '' &&
                          result.split(searchInput)[0]}
                        <span
                          className='r-medium'
                          style={{ color: 'var(--main-blue)' }}
                        >
                          {searchInput}
                        </span>
                        {result.split(searchInput)[1] !== '' &&
                          result.split(searchInput)[1]}
                      </p>
                      {!isLoading &&
                        searchResults &&
                        searchResults.length >= 10 && <div ref={setTarget} />}
                    </div>
                  ))
                ) : (
                  <p
                    className='r-medium'
                    style={{ textAlign: 'center', marginTop: '60px' }}
                  >
                    검색 결과가 없습니다.
                  </p>
                )}
              </div>

              <p
                className='r-medium'
                style={{
                  color: 'var(--main-skyblue)',
                  marginLeft: 'auto',
                  marginTop: 'auto',
                  paddingTop: '10px',
                }}
                onClick={() => setSearching(false)}
              >
                닫기
              </p>
            </div>
          )}
        </div>
      </div>

      <div>
        <div className={styles.caltext}>
          <p className='r-large'>얼마나 먹었나요?</p>
          <p className='r-super'>
            {focusing.totalCapacity
              ? focusing.counts * focusing.totalCapacity
              : 0}
            g
          </p>
        </div>
        <div className={styles.calinput}>
          <img
            className={styles.calbtn}
            src='/icons/minusicon.png'
            alt='-'
            onClick={decrement}
          />
          <p className='s-big'>{focusing.counts}</p>
          <img
            className={styles.calbtn}
            src='/icons/plusicon.png'
            alt='+'
            onClick={increment}
          />
        </div>
      </div>
    </div>
  );
};

export default RecordEditDetail;

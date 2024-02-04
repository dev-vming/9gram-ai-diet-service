import ButtonCommon from '@components/UI/ButtonCommon';
import InputCommon from '../../UI/InputCommon';
import styles from './addphotosearch.module.css';
import { useEffect, useMemo, useState } from 'react';
import AddTag from './AddTag';
import DeleteTag from './DeleteTag';
import { useNavigate, useParams } from 'react-router-dom';
import useApi from '@hooks/useApi';
import getFoodId from '@utils/getFoodId';

interface FoodInfo {
  foodInfoId: string;
  foodName: string;
}

interface Result {
  data: FoodInfo[];
}

const AddPhotoSearch = () => {
  const navigate = useNavigate();
  const params = useParams();
  const date = params.date;
  const mealTime = params.mealTime;

  const [searchInput, setSearchInput] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [tags, setTags] = useState<string[]>([]);
  const [searchResults, setSearchResults] = useState(['']);
  const [foodInfo, setFoodInfo] = useState<FoodInfo[]>([]);

  const { result, trigger } = useApi<Result>({
    path: `/food-info/foods?keyword=${searchInput}`,
  });

  const handleClick = () => {
    if (!searchInput) return;
    if (searchResults.length >= 10) return;
    trigger({});
  };

  useEffect(() => {
    if (!result) return;
    const newResults = result.data.map((food: FoodInfo) => (
      food.foodName.includes('_')
        ? food.foodName.split('_')[1]
        : food.foodName
  ));
    setSearchResults(newResults);
    setFoodInfo(result.data);
    setIsSearching(true);
  }, [result])

  

  const tagsArr = useMemo(() =>
    tags.map((tag) => ({
      foodName: tag,
      XYCoordinate: [],
      counts: 1,
      foodInfoId: getFoodId(tag,foodInfo,searchInput)})) || [], [tags])
  
  const data = {
    imgUrl: undefined,
    foods: tagsArr,
  }

  const addMeal = () => {
    navigate(`/record/${date}/${mealTime}/edit`, {state: data });
  };

  return (
    <div className={styles.container}>
      <div className={styles.searchbox}>
        <InputCommon
          size='medium'
          variant='default'
          value={searchInput}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
            setSearchInput(e.target.value);
            setIsSearching(false);
            setSearchResults([]);
            setFoodInfo([]);
          }}
        />
        <ButtonCommon
          size='small'
          variant='default-active'
          onClickBtn={handleClick}
        >
          검색
        </ButtonCommon>
      </div>
      <div className={styles.resultbox}>
        {isSearching && (
          <AddTag
            searchInput={searchInput}
            tags={tags}
            setTags={setTags}
            setIsSearching={setIsSearching}
            searchResults={searchResults}
            foodInfo={foodInfo}
            setSearchResults={setSearchResults}
            setFoodInfo={setFoodInfo}
          />
        )}
      </div>
      <div className={styles.tagbox}>
        {tags && <DeleteTag tags={tags} setTags={setTags} />}
      </div>
      <div className={styles.btnbox}>
        {tags.length ? (
          <ButtonCommon
            size='big'
            variant='default-active'
            onClickBtn={addMeal}
          >
            식단 추가하기
          </ButtonCommon>
        ) : (
          <ButtonCommon size='big' variant='default' disabled={true}>
            식단 추가하기
          </ButtonCommon>
        )}
      </div>
    </div>
  );
};

export default AddPhotoSearch;

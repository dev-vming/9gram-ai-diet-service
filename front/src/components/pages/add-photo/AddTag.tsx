import useApi from '@hooks/useApi';
import styles from './addtag.module.css';
import useIntersect from '@hooks/useIntersect';
import { useEffect, useMemo, useState } from 'react';

interface FoodInfo {
  foodInfoId: string;
  foodName: string;
}

interface Result {
  data: FoodInfo[];
}

interface Props {
  searchInput: string;
  tags: string[];
  setTags: React.Dispatch<React.SetStateAction<string[]>>;
  setIsSearching: React.Dispatch<React.SetStateAction<boolean>>;
  searchResults: string[];
  foodInfo: { foodName: string; foodInfoId: string }[];
  setSearchResults: React.Dispatch<React.SetStateAction<string[]>>;
  setFoodInfo: React.Dispatch<React.SetStateAction<FoodInfo[]>>;
}

const AddTag = ({
  searchInput,
  tags,
  setTags,
  setIsSearching,
  searchResults,
  foodInfo,
  setSearchResults,
  setFoodInfo,
}: Props) => {
  const [isLoading, setIsLoading] = useState(false);
  const lastFoodId =
    foodInfo.length > 0 && foodInfo[foodInfo.length - 1].foodInfoId;

  const isVisibleObserverTarget = useMemo(() => {
    return !isLoading && searchResults?.length >= 10;
  }, [isLoading, searchResults]);

  const { result, trigger } = useApi<Result>({
    path: `/food-info/foods?keyword=${searchInput}&lastFoodId=${lastFoodId}`,
  });

  const onIntersect: IntersectionObserverCallback = async (
    [entry],
    observer
  ) => {
    if (entry.isIntersecting) {
      if (result?.data?.length === 0) return;
      setIsLoading(true);
      observer.unobserve(entry.target);
      await trigger({});
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
    if (!result) return;
    const newResults = result.data.map((food: FoodInfo) =>
      food.foodName.includes('_') ? food.foodName.split('_')[1] : food.foodName
    );
    setSearchResults(searchResults.concat(newResults));
    setFoodInfo(foodInfo.concat(result.data));
  }, [result]);

  const addTag = (e: React.MouseEvent<HTMLParagraphElement, MouseEvent>) => {
    if (!e.currentTarget.textContent) {
      throw new Error('지정할 태그 내용이 없습니다!');
    }
    tags.includes(e.currentTarget.textContent)
      ? alert('이미 추가된 태그입니다!')
      : setTags([...tags, e.currentTarget.textContent]);
  };

  return (
    <>
      <div className={styles.result}>
        <div style={{ overflow: 'auto' }}>
          {searchInput && searchResults.length > 0 ? (
            searchResults.map((result, index) => (
              <div key={index}>
                <p
                  onClick={(e) => addTag(e)}
                  className='r-big'
                  style={{ marginBottom: '5px' }}
                >
                  {result.split(searchInput)[0] !== '' &&
                    result.split(searchInput)[0]}
                  <span className={styles.searchtext}>{searchInput}</span>
                  {result.split(searchInput)[1] !== '' &&
                    result.split(searchInput)[1]}
                </p>
                {isVisibleObserverTarget && (
                  <div ref={setTarget} />
                )}
              </div>
            ))
          ) : (
            <p
              className='r-medium'
              style={{ textAlign: 'center', marginTop: '75px' }}
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
          onClick={() => setIsSearching(false)}
        >
          닫기
        </p>
      </div>
    </>
  );
};

export default AddTag;

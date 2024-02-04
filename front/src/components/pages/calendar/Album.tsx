import { useNavigate } from 'react-router-dom';
import { useCalendarContext } from './Calendar';
import classes from './album.module.css';
import { useEffect, useState } from 'react';
import useIntersect from '@hooks/useIntersect';
import AlbumBody from './AlbumBody';
import useApi from '@hooks/useApi';
export type MealType = '아침' | '점심' | '저녁' | '간식';

export interface AlbumArrType {
  date: string;
  dateArr: [MealType, number, string][];
}

export const returnWithZero = (num: string | number) => {
  return Number(num) < 10 ? `0${num}` : num;
};

//현재 연도와 월 가져오기
type AlbumApiResponse = {
  data: AlbumArrType[];
};
const Album = () => {
  const { thisYear, thisMonth } = useCalendarContext();
  const [isLoading, setIsLoading] = useState(false);
  const [isFirst, setIsFirst] = useState(true);
  const [albumArr, setAlbumArr] = useState<AlbumArrType[]>([]);
  const [page, setPage] = useState(1);
  const { trigger, result } = useApi<AlbumApiResponse>({
    path: `/cumulative-record/month?month=${thisYear}-${returnWithZero(thisMonth)}-01&page=${page}`,
  });
  const navigate = useNavigate();

  const onClickCards = (val: string) => {
    console.log(val);
    navigate(
      `/record/${thisYear}-${returnWithZero(thisMonth)}-${returnWithZero(val)}`
    );
  };

  useEffect(() => {
    if (!isFirst) return;
    trigger({});
    setPage((prev) => prev + 1);
    setIsFirst(false);
  }, []);

  useEffect(() => {
    if (!result) return;
    setAlbumArr((prev) => [...prev, ...result.data]);
  }, [result]);

  const onIntersect: IntersectionObserverCallback = async (
    [entry],
    observer
  ) => {
    if (entry.isIntersecting) {
      if (!result) return;
      if (result.data.length === 0) return;
      setIsLoading(true);
      observer.unobserve(entry.target);
      await trigger({});
      observer.observe(entry.target);
      setIsLoading(false);
      setPage((prev) => prev + 1);
    }
  };

  const { setTarget } = useIntersect({
    root: null,
    rootMargin: '100px',
    threshold: 1,
    onIntersect,
  });

  return (
    <div className={classes.wrapper}>
      {albumArr &&
        albumArr?.map((albumDay, idx) => (
          <AlbumBody
            key={`body-${idx}`}
            idx={idx}
            setTarget={setTarget}
            albumDay={albumDay}
            onClickCards={onClickCards}
            isLoading={isLoading}
          />
        ))}
      {!isLoading && <div ref={setTarget} />}
    </div>
  );
};

export default Album;

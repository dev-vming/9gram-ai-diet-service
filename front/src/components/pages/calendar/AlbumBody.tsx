import { getMealsNumber } from '@utils/getMealNum';
import { AlbumArrType, returnWithZero } from './Album';
import AlbumCell from './AlbumCell';
import { useCalendarContext } from './Calendar';
import classes from './album.module.css';
const AlbumBody = ({
  idx,
  albumDay,
  onClickCards,
}: {
  idx: number;
  setTarget: React.Dispatch<React.SetStateAction<HTMLElement | null>>;
  albumDay: AlbumArrType;
  onClickCards: (val: string) => void;
  isLoading: boolean;
}) => {
  const { thisYear, thisMonth } = useCalendarContext();

  return (
    <div key={`album-${idx}`} className={classes.date}>
      <div
        className={`b-regular`}
      >{`${thisYear}.${returnWithZero(thisMonth)}.${returnWithZero(albumDay?.date)}`}</div>
      <div className={classes.cards}>
        {albumDay?.dateArr &&
          albumDay?.dateArr.map((arr, idx) => (
            <div
              key={`album-${idx}`}
              onClick={() =>
                onClickCards(
                  `${returnWithZero(albumDay.date)}/${getMealsNumber[arr[0]]}`
                )
              }
            >
              <AlbumCell
                arr={arr}
                idx={idx}
                key={`album-${idx}`}
              />
            </div>
          ))}
      </div>
    </div>
  );
};

export default AlbumBody;

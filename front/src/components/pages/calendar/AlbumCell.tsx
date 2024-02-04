import classes from './album.module.css';
import { MealType } from './Album';
import { getNumberMeal } from '@utils/getMealNum';

const AlbumCell = ({
  arr,
  idx,
}: {
  arr: [MealType, number, string];
  idx: number;
}) => {
  return (
    <>
      <div className={classes['meal-img']}>
        {arr[2]
          ? (<img src={arr[2]} alt='img' style={{ objectFit: "cover", filter: "brightness(0.7)" }} />)
          : (<img src='/images/9gram_logo.png' alt='img' style={{ objectFit: "cover", filter: "brightness(0.7)"}} />)}
        
        <div className={classes['meal-card']} key={`date-${idx}`}>
            <>
              <div className='b-regular' style={{color:'white'}}>
                {getNumberMeal[arr[0]]}
              </div>
              <div className='b-medium' style={{color:'white'}}>
                {Math.round(arr[1])} kcal
              </div>
            </>
        </div>
      </div>
    </>
  );
};

export default AlbumCell;

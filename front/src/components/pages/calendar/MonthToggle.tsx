import { ReactElement, ReactNode, useEffect } from 'react';
import { useCalendarContext } from './Calendar';
//선택하면 나타나게, 월을 선택하면 사라지게.

import classes from './monthToggle.module.css';

interface MonthTogglePropsType {
  children?: ReactElement[] | ReactElement | string;
}

const MonthToggle = ({ children }: MonthTogglePropsType) => {
  const { thisYear, thisMonth, setShowSelect, showSelect } =
    useCalendarContext();

  const onClickBtn = () => {
    setShowSelect((prev) => !prev);
  };

  return (
    <>
      <div className={classes['wrapper']} onClick={onClickBtn}>
        <div className='b-large'>{`${thisYear}.${thisMonth}`}</div>
        <div className='caret'></div>
      </div>
      {showSelect && children}
    </>
  );
};

export default MonthToggle;

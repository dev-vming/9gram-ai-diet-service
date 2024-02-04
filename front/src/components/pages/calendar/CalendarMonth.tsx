import { ReactElement, ReactNode, useState } from 'react';
import { useCalendarContext } from './Calendar';
import classes from './calendarMonth.module.css';

interface CalendarMonthProps {
  children?: string | ReactElement[] | ReactElement;
}

const CalendarMonth = ({ children }: CalendarMonthProps) => {
  const { setThisYear, setThisMonth, thisYear, setShowSelect } =
    useCalendarContext();
  const [selectedYear, setSelectedYear] = useState(thisYear);

  const onClickMonth = (e: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    const clickedElementId = Number(
      (e.target as HTMLDivElement).id.split('-')[1]
    );
    console.log(clickedElementId);
    setThisYear(selectedYear);
    setThisMonth(clickedElementId);
    setShowSelect((prev) => !prev);
  };

  return (
    <div className={classes.wrapper}>
      <div className={classes.header}>
        <p onClick={() => setSelectedYear((prev) => prev - 1)}>&lt;</p>
        <div>{selectedYear}</div>
        <p onClick={() => setSelectedYear((prev) => prev + 1)}>&gt;</p>
      </div>
      <div className={classes.monthWrapper}>
        {new Array(12).fill(0).map((_, idx) => (
          <div
            onClick={onClickMonth}
            id={`month-${idx + 1}`}
            key={`month-${idx + 1}`}
          >
            {idx + 1}
          </div>
        ))}
      </div>
    </div>
  );
};

export default CalendarMonth;

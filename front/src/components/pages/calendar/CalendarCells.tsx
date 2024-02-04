import classes from './calendarbody.module.css';

interface CalendarCellsProps {
  thisMonthFirstDay: number;
  thisMonthTotal: number;
  idx: number;
  selectedIndex: number;
  setSelectedIndex: React.Dispatch<React.SetStateAction<number>>;
  cumulative_cal_DateArr?: {
    existedDate: number[] | any[];
    totalCalData: number[] | any[];
  };
  targetCalories: number;
}

const CalendarCells = ({
  thisMonthFirstDay,
  thisMonthTotal,
  idx,
  selectedIndex,
  setSelectedIndex,
  targetCalories,
  cumulative_cal_DateArr,
}: CalendarCellsProps) => {
  //눈에 보이는 실제 날짜
  const getDayNumber = (idx: number): number => {
    return idx - thisMonthFirstDay + 1;
  };

  //달력 arr중 날짜가 있는 셀의 인덱스인지
  const isDate =
    idx >= thisMonthFirstDay && idx < thisMonthTotal + thisMonthFirstDay;

  //현재 날짜의 existedDate의 index
  const existedDateIndex =
    cumulative_cal_DateArr &&
    cumulative_cal_DateArr['existedDate'] &&
    cumulative_cal_DateArr['existedDate']?.findIndex(
      (el) => el === getDayNumber(idx)
    );

  //날짜가 있는지에 따른 기본 셀 클래스
  const cellStyle = isDate
    ? `${classes['day-wrapper']} b-small`
    : ` ${classes['cal-circle']} b-small`;

  //상태에 따른 className
  const colorCls =
    cumulative_cal_DateArr &&
    existedDateIndex &&
    cumulative_cal_DateArr['totalCalData'] &&
    cumulative_cal_DateArr?.['totalCalData'][existedDateIndex] > targetCalories
      ? 'over-eat'
      : 'moderate';

  //목표 칼로리 비교 색깔 구분 적당 vs 과식
  const colorFonts =
    colorCls === 'over-eat'
      ? `${classes[`font-over-eat`]}`
      : classes[`font-moderate`];

  const selectedCls =
    selectedIndex === idx - thisMonthFirstDay + 1
      ? ` ${classes['selected']}`
      : ``;

  //calory데이터 있는지 여부
  const existedCalData = cumulative_cal_DateArr?.['existedDate']?.includes(
    getDayNumber(idx)
  );
  //데이터 있으면 타겟 칼로리 대비 섭취량에 따라 색상다르게 적용(colorCls)
  const existedCalDataCls = existedCalData
    ? `${classes[`${colorCls}`]} b-small` + selectedCls
    : `${classes['cal-circle']} b-small` + selectedCls;

  //선택된 font스타일
  const selectedFontsStyle =
    selectedCls !== '' ? ` ${classes['font-selected']}` : '';

  const onClickDate = (e: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    const clickedElementId = Number(
      (e.target as HTMLDivElement).id.split('-')[1]
    );
    //기존에 index가 동일하면 삭제
    setSelectedIndex((prev) => {
      return prev === clickedElementId ? NaN : clickedElementId;
    });
  };

  return (
    <div key={`$cal-${idx}`} className={cellStyle}>
      <div
        key={`$date-${idx}`}
        className={existedCalDataCls}
        id={isDate ? `idx-${getDayNumber(idx)}` : undefined}
        onClick={onClickDate}
      >
        {isDate && getDayNumber(idx)}
      </div>

      {existedCalData &&
        cumulative_cal_DateArr &&
        typeof existedDateIndex === 'number' &&
        existedDateIndex > -1 && (
          <p
            key={`p-${idx}`}
            className={`r-regular ${colorFonts + selectedFontsStyle}`}
          >
            {`+${cumulative_cal_DateArr['totalCalData'][existedDateIndex]}`}
          </p>
        )}
    </div>
  );
};

export default CalendarCells;

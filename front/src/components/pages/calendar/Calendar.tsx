import CarlendarHeader from './CalendarHeader';
import CalendarBody from './CalendarBody';
import { createContext, useEffect } from 'react';
import { Dispatch, ReactElement, useContext, useState } from 'react';
import getDates from '../../../utils/getDates';
import useCachingApi from '@hooks/useCachingApi';

const CalendarContext = createContext<
  | {
      isAlbum: boolean;
      setIsAlbum: Dispatch<React.SetStateAction<boolean>>;
      thisYear: number;
      setThisYear: Dispatch<React.SetStateAction<number>>;
      thisMonth: number;
      setThisMonth: Dispatch<React.SetStateAction<number>>;
      thisDay: number;
      setThisDay: Dispatch<React.SetStateAction<number>>;
      selectedIndex: number;
      setSelectedIndex: Dispatch<React.SetStateAction<number>>;
      showSelect: boolean;
      setShowSelect: Dispatch<React.SetStateAction<boolean>>;
    }
  | undefined
>(undefined);

const CalendarProvider = ({
  children,
}: {
  children: ReactElement[] | ReactElement;
}) => {
  const {
    thisYear: yearNow,
    thisMonth: monthNow,
    thisDay: dayNow,
  } = getDates();

  const [thisYear, setThisYear] = useState(yearNow);
  const [thisMonth, setThisMonth] = useState(Number(monthNow));
  const [thisDay, setThisDay] = useState(Number(dayNow));
  const [isAlbum, setIsAlbum] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(NaN);
  const [showSelect, setShowSelect] = useState(false);

  return (
    <CalendarContext.Provider
      value={{
        thisYear,
        setThisYear,
        thisMonth,
        setThisMonth,
        thisDay,
        setThisDay,
        selectedIndex,
        setSelectedIndex,
        isAlbum,
        setIsAlbum,
        showSelect,
        setShowSelect,
      }}
    >
      {children}
    </CalendarContext.Provider>
  );
};

export const useCalendarContext = () => {
  const context = useContext(CalendarContext);
  if (context === undefined) {
    throw new Error(
      'useCalendarContext는 CalendarProvider내부에서만 사용할 수 있습니다'
    );
  }
  return context;
};

const Calendar = () => {
  return (
    <>
      <CalendarProvider>
        <CarlendarHeader />
        <CalendarBody />
      </CalendarProvider>
    </>
  );
};

export default Calendar;

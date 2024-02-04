import { Next } from '@assets/Next';
import styles from '@components/pages/ai-analyze/drawer.module.css';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { mapGoaltoMsg } from '../my-page/mapMsg';

interface Props {
  type: string;
  tag: string;
  option: { goal: string; targetCalories: number } | undefined;
}

const Option = ({ type, tag, option }: Props) => {
  const [showOption, setShowOption] = useState(true);
  const [icon, setIcon] = useState(false);
  const [detail, setDetail] = useState(false);
  const goalMsg = option && mapGoaltoMsg[Number(option.goal)];
  useEffect(() => {
    if (type === '식단추천' && tag === '오늘은 맛있는 걸로 추천받을래!') {
      setShowOption(false);
    } else if (type === '식단평가') {
      setIcon(true);
    } else {
      setDetail(true);
    }
  }, [type]);

  const navigate = useNavigate();

  return (
    <>
      {showOption && (
        <div className={styles.option_wrapper}>
          {icon && (
            <div className={styles.option} onClick={() => navigate('/')}>
              <div className='s-big'>식단 보러 가기</div>
              <Next />
            </div>
          )}
          {option && detail && (
            <div className={styles.detail}>
              <div className={styles.option}>
                <p className='s-big'>목표</p>
                <p className={`${styles.option_goal} s-large`}>{goalMsg}</p>
              </div>
              <div className={styles.option}>
                <p className='r-big'>목표 칼로리</p>
                <p className='r-large'>{option.targetCalories}kcal</p>
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
};

export default Option;

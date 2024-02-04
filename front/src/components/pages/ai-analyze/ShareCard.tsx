import { useLocation, useNavigate } from 'react-router-dom';
import styles from '@components/pages/ai-analyze/drawer.module.css';
import { Next } from '@assets/Next';
import useApi from '@hooks/useApi';
import { useEffect, useState } from 'react';

const typeType: Record<string, string> = {
  식단추천: styles.recommend,
  식단평가: styles.analyze,
  목표추천: styles.goal,
};
const tagType: Record<string, string> = {
  식단추천: styles.recommend_tag,
  식단평가: styles.analyze_tag,
  목표추천: styles.goal_tag,
};

interface Props {
  feedbackId: string;
  feedbackDate: string;
  questionType: string;
  question: string;
  feedback: string;
  option: { goal: string; calorie: number };
}
interface PropsReturnType {
  data: Props;
}

const ShareCard = () => {
  const location = useLocation();
  const id = location.pathname.split('/')[2];

  const { trigger, result, reqIdentifier, loading, error } =
    useApi<PropsReturnType>({
      method: 'get',
      path: `/feedback?feedbackId=${id}`,
      shouldInitFetch: false,
    });

  const triggerData = async () => {
    await trigger({
      applyResult: true,
      isShowBoundary: true,
    });
  };

  const [data, setData] = useState({
    feedbackId: '',
    feedbackDate: '',
    questionType: '',
    question: '',
    feedback: '',
    option: { goal: '', calorie: 0 },
  });

  useEffect(() => {
    triggerData();
  }, []);

  useEffect(() => {
    if (result?.data) {
      setData(result?.data);
    }
  }, [result?.data]);

  const navigate = useNavigate();

  const newDate = data.feedbackDate.split('-').join('.');

  return (
    <>
      <div className={styles.drawer_wrapper}>
        <div className={styles.detail_wrapper}>
          <div className={`${styles.date} b-regular`}>{newDate}</div>
          <div
            className={`${styles.type} ${typeType[data.questionType]} s-regular`}
          >
            {data.questionType}
          </div>
          <div
            className={`${styles.tag} ${tagType[data.questionType]} b-small`}
          >
            {data.question}
          </div>
          <div className={styles.option_wrapper}>
            <div className={styles.option} onClick={() => navigate('/join')}>
              <div className='s-big'>로그인하고 내 식단 평가 받으러 가기</div>
              <Next />
            </div>
          </div>
          <div className={`${styles.share_text} r-big`}>{data.feedback}</div>
        </div>
      </div>
    </>
  );
};

export default ShareCard;

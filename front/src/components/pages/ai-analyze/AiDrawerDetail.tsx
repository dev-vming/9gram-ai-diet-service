import styles from '@components/pages/ai-analyze/drawer.module.css';
import Option from './Option';
import { Share } from '@assets/Share';
import { DeleteBox } from '@assets/DeleteBox';
import { useEffect, useState } from 'react';
import MiniToast from './MiniToast';
import useApi from '@hooks/useApi';
import { useNavigate, useSearchParams } from 'react-router-dom';

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
  option: { goal: string; targetCalories: number };
}
interface PropsReturnType {
  data: Props;
}

const handleCopyClipBoard = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    console.log(err);
  }
};

const AiDrawerDetail = () => {
  const [shareToast, setShareToast] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();
  const id = searchParams.get('feedbackId');
  const location = window.location.protocol + '//' + window.location.host;

  const navigate = useNavigate();

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
  const triggerDeleteData = async () => {
    await trigger({
      method: 'delete',
      path: `/feedback?feedbackId=${id}`,
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
    option: { goal: '', targetCalories: 0 },
  });

  useEffect(() => {
    triggerData();
  }, []);

  useEffect(() => {
    if (result?.data) {
      setData(result?.data);
    }
  }, [result?.data]);

  const handleShare = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    handleCopyClipBoard(`${location}/share/${data?.feedbackId}`); // 나중에 배포 url로 변경 필요!
    setShareToast(true);
  };
  const handleDelete = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    triggerDeleteData();
    navigate(-1);
  };

  const newDate = data.feedbackDate && data.feedbackDate.split('-').join('.');

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
          {data.question !== '목표추천' && (
            <div
              className={`${styles.tag} ${tagType[data.questionType]} b-small`}
            >
              {data.question}
            </div>
          )}
          <Option
            type={data.questionType}
            tag={data.question}
            option={data.option && data.option}
          />
          <div className={`${styles.detail_text} r-big`}>{data.feedback}</div>
          <div className={styles.button_wrapper}>
            <button className={`r-small`} onClick={handleShare}>
              <Share />
              공유하기
            </button>
            <button className={`r-small`} onClick={handleDelete}>
              <DeleteBox />
              삭제하기
            </button>
          </div>
        </div>
      </div>
      {shareToast && (
        <MiniToast setToast={setShareToast} text='답변 URL이 복사되었습니다.' />
      )}
    </>
  );
};

export default AiDrawerDetail;

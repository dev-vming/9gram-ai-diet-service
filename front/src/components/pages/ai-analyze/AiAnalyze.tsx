import styles from '@components/pages/ai-analyze/aianalyze.module.css';
import { useEffect, useRef, useState } from 'react';
import BotBox from './BotBox';
import UserBox from './UserBox';
import { questionData } from './QuestionData';
import { useNavigate } from 'react-router-dom';
import useCachingApi from '@hooks/useCachingApi';
import useApi from '@hooks/useApi';

type RecordList = {
  feedbackId: string;
  feedbackDate: string;
  questionType: string;
  question: string | undefined;
  feedback: string;
}[];
interface AskData {
  questionType: string;
  question: string;
}

interface Context {
  type: {
    questionType: string;
    question: string | undefined;
  };
  text: string;
  button: { type: string; text: string }[];
}
type QuestionList = {
  date: string;
  questionIdx: string;
  context: Context;
  answer: string;
  feedbackId: string;
}[];
const now = new Date();
const nowYear = now.getFullYear();
const nowMonth = now.getMonth();
const nowDate = now.getDate();
const lastLastDayIndex = new Date(nowYear, nowMonth, 0).getDate(); // 저번달 마지막 날짜

const AiAnalyze = () => {
  const [recordText, setRecordText] = useState(false);
  const todayDate = `${nowYear}-${nowMonth + 1 >= 10 ? nowMonth + 1 : `0${nowMonth + 1}`}-${nowDate >= 10 ? nowDate : `0${nowDate}`}`;
  const sixDays = nowDate - 6; // 6일 전
  let startDate = '';
  if (nowDate - 6 <= 0) {
    const newSixDays = lastLastDayIndex + sixDays;
    startDate += `${nowYear}-${nowMonth >= 10 ? nowMonth : `0${nowMonth}`}-${newSixDays >= 10 ? newSixDays : `0${newSixDays}`}`;
  } else {
    startDate += `${nowYear}-${nowMonth + 1 >= 10 ? nowMonth + 1 : `0${nowMonth + 1}`}-${sixDays >= 10 ? sixDays : `0${sixDays}`}`;
  }

  const { trigger, result }: { trigger: any; result: any } = useCachingApi({
    path: `/feedback?startDate=${startDate}&date=${todayDate}`,
    gcTime: 3000,
  });

  const triggerData = async () => {
    await trigger({});
  };

  useEffect(() => {
    triggerData();
  }, []);

  useEffect(() => {
    if (result?.data && result?.data.data.length !== 0) {
      setRecordText(true);
    }
  }, [result?.data]);

  const [chats, setChats] = useState<QuestionList>([
    {
      date: todayDate,
      questionIdx: '1',
      context: questionData['1'],
      answer: '3',
      feedbackId: '',
    },
  ]);

  const [answerIdx, setAnswerIdx] = useState(3);
  const [questionIdx, setQuestionIdx] = useState('1');
  const [prevQuestionIdx, setPrevQuestionIdx] = useState('1');

  const formatRecord = (recordList: RecordList) => {
    const questionList: QuestionList = [];
    recordList.forEach((record, idx) => {
      const recordQuestionIdx = Object.keys(questionData).find(
        (key) =>
          questionData[key].type.questionType === record.questionType &&
          questionData[key].type.question === record.question
      );
      const splitIdx = recordQuestionIdx?.split('-');
      const questionIdxList = splitIdx?.map(
        (num, idx) =>
          (splitIdx[idx - 2] ? splitIdx[idx - 2] + '-' : '') +
          (splitIdx[idx - 1] ? splitIdx[idx - 1] + '-' : '') +
          num
      );
      const question = questionIdxList?.map((num, qIdx) => {
        let context = {};
        if (num.length === 5 || num === '1-3') {
          const oldContext = questionData[num].text.split('\n');
          oldContext.splice(oldContext.length - 1, 0, record.feedback);
          const newContext = oldContext.join('\n');
          context = {
            type: {
              questionType: questionData[num].type.questionType,
              question: questionData[num].type.question,
            },
            text: newContext,
            button: questionData[num].button,
          };
        } else {
          context = questionData[num];
        }

        if (
          questionList.length === 0 ||
          record.feedbackDate === questionList[questionList.length - 1].date
        ) {
          if (num === '1') {
            if (questionList.length !== 0 && idx !== 0) {
              return {
                date: record.feedbackDate,
                questionIdx: num,
                context: context,
                answer: '다른 질문도 할래!',
                feedbackId: record.feedbackId,
              };
            } else if (idx !== 0) {
              return {
                date: record.feedbackDate,
                questionIdx: num,
                context: context,
                answer: '다른 질문도 할래!',
                feedbackId: record.feedbackId,
              };
            } else {
              return {
                date: record.feedbackDate,
                questionIdx: num,
                context: context,
                answer: '3',
                feedbackId: record.feedbackId,
              };
            }
          } else {
            return {
              date: record.feedbackDate,
              questionIdx: num,
              context: context,
              answer: splitIdx
                ? questionData[questionIdxList[qIdx - 1]]?.button[
                    Number(splitIdx[qIdx]) - 1
                  ].text
                : '3',
              feedbackId: record.feedbackId,
            };
          }
        } else if (num === '1') {
          return {
            date: record.feedbackDate,
            questionIdx: num,
            context: context,
            answer: '3',
            feedbackId: record.feedbackId,
          };
        } else {
          return {
            date: record.feedbackDate,
            questionIdx: num,
            context: context,
            answer: splitIdx
              ? questionData[questionIdxList[qIdx - 1]]?.button[
                  Number(splitIdx[qIdx]) - 1
                ].text
              : '3',
            feedbackId: record.feedbackId,
          };
        }
      });
      console.log(question);
      if (question) {
        questionList.push(...(question as QuestionList));
      }
    });
    return questionList;
  };

  useEffect(() => {
    if (recordText) {
      const questionList = formatRecord(result?.data.data);
      console.log(questionList);
      if (questionList[questionList.length - 1].date !== todayDate) {
        setChats((prev) => [
          ...questionList,
          {
            date: todayDate,
            questionIdx: '1',
            context: questionData['1'],
            answer: '3',
            feedbackId: '',
          },
        ]);
      } else if (chats.length !== 1) {
        setChats((prev) => [
          {
            date: todayDate,
            questionIdx: '1',
            context: questionData['1'],
            answer: '3',
            feedbackId: '',
          },
          ...questionList,
        ]);
        setQuestionIdx(questionList[questionList.length - 1].questionIdx);
      } else {
        setChats((prev) => [...questionList]);
        setQuestionIdx(questionList[questionList.length - 1].questionIdx);
      }
    } else if (chats.length === 1) {
      setChats((prev) => [
        {
          date: todayDate,
          questionIdx: '1',
          context: questionData['1'],
          answer: '3',
          feedbackId: '',
        },
      ]);
    } else {
      setChats((prev) => [...prev]);
    }
  }, [recordText]);

  const navigate = useNavigate();

  const {
    trigger: askTrigger,
    result: askResult,
  }: { trigger: any; result: any } = useApi({
    method: 'post',
    path: `/feedback?date=${todayDate}`,
    shouldInitFetch: false,
  });

  const askGPT = async (askData: AskData) => {
    await askTrigger({
      applyResult: true,
      isShowBoundary: true,
      data: askData,
    });
  };
  const [gptAnswer, setGptAnswer] = useState('');
  const [gptId, setGptId] = useState('');

  const handleOnClick = (
    e: React.MouseEvent<HTMLButtonElement>,
    idx: number
  ) => {
    if (questionData[questionIdx].button[idx].type === 'follow-up') {
    } else if (questionData[questionIdx].button[idx].type === 'navigate') {
      if (questionData[questionIdx].type.questionType === '식단추천') {
        navigate(`/record/${todayDate}`);
      } else if (questionData[questionIdx].type.questionType === '목표추천') {
        navigate('/my-page');
      } else {
        navigate('/home');
      }
    } else if (questionData[questionIdx].button[idx].type === 'ai') {
      const askData = questionData[questionIdx + '-' + String(idx + 1)]
        .type as AskData;
      askGPT(askData);
    }
    setAnswerIdx(idx);
    setQuestionIdx((prev) => {
      setPrevQuestionIdx(prev);
      return prev.length === 5 || prev === '1-3'
        ? '1'
        : prev + '-' + String(idx + 1);
    });
  };

  useEffect(() => {
    if (askResult) {
      setGptAnswer(askResult?.data.feedback);
      setGptId(askResult?.data.feedbackId);
    }
  }, [askResult]);

  useEffect(() => {
    if (
      (answerIdx < 3 || gptId.length > 0) &&
      questionIdx.length !== 5 &&
      questionIdx !== '1-3'
    ) {
      setChats((prev) => [
        ...prev,
        {
          date: todayDate,
          questionIdx: questionIdx,
          context: questionData[questionIdx],
          answer: questionData[prevQuestionIdx].button[answerIdx].text,
          feedbackId: gptId,
        },
      ]);
    }
  }, [questionIdx]);

  useEffect(() => {
    if (questionIdx.length === 5 || questionIdx === '1-3') {
      const newContext = questionData[questionIdx];
      const oldText = questionData[questionIdx].text.split('\n');
      oldText.splice(oldText.length - 1, 0, gptAnswer);
      const newText = oldText.join('\n');
      newContext.text = newText;
      setChats((prev) => [
        ...prev,
        {
          date: todayDate,
          questionIdx: questionIdx,
          context: newContext,
          answer: questionData[prevQuestionIdx].button[answerIdx].text,
          feedbackId: gptId,
        },
      ]);
    }
  }, [gptId]);

  const chatEndRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    const timeoutId = setTimeout(() => {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 900);
    return () => clearTimeout(timeoutId);
  }, [chats]);

  console.log(chats);

  return (
    <div className={styles.main_wrapper}>
      <>
        {chats.map((chat, idx) => (
          <div key={`chat-${idx}`} className={styles.chats_wrapper}>
            {(idx === 0 || (idx > 1 && chat.date !== chats[idx - 1].date)) && (
              <div className={`${styles.date} s-regular`}>{chat.date}</div>
            )}
            {chat.answer !== '3' && <UserBox text={chat.answer} />}
            {/* {idx !== 0 ? ( */}
            <BotBox
              toSave={chat.context.type.question ? true : false}
              text={chat.context.text}
              button={chat.context.button}
              feedbackId={chat.feedbackId}
              handleOnClick={handleOnClick}
              disabled={
                chats.length < 2 || idx === chats.length - 1 ? false : true
              }
            />
            {/* ) : (
              <></>
            )} */}
          </div>
        ))}
        <div ref={chatEndRef}></div>
      </>
    </div>
  );
};

export default AiAnalyze;

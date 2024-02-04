import { Scrap } from '@assets/Scrap';
import styles from '@components/pages/ai-analyze/box.module.css';
import ButtonCommon from '@components/UI/ButtonCommon';
import useApi from '@hooks/useApi';
import { useEffect, useState } from 'react';
import MiniToast from './MiniToast';

interface Props {
  toSave: boolean;
  text: string;
  button: { type: string; text: string }[];
  feedbackId: string;
  handleOnClick: (e: React.MouseEvent<HTMLButtonElement>, idx: number) => void;
  disabled: boolean;
}

const BotBox = ({
  toSave,
  text,
  button,
  feedbackId,
  handleOnClick,
  disabled,
}: Props) => {
  const [scrap, setScrap] = useState(false);
  const [clicked, setClicked] = useState(false);
  const [toast, setToast] = useState(false);

  const { trigger, result, reqIdentifier, loading, error } = useApi({
    method: 'get',
    path: `/feedback/save?feedbackId=${feedbackId}`,
    shouldInitFetch: false,
  });

  const saveScrap = async () => {
    await trigger({
      applyResult: true,
      isShowBoundary: true,
    });
  };
  const deleteScrap = async () => {
    await trigger({
      method: 'delete',
      path: `/feedback?feedbackId=${feedbackId}`,
      applyResult: true,
      isShowBoundary: true,
    });
  };

  const handleScrap = (e: React.MouseEvent<HTMLDivElement>) => {
    setScrap(!scrap);
    if (!scrap) {
      setToast(true);
      saveScrap();
    } else {
      setToast(false);
      deleteScrap();
    }
  };

  const [delayed, setDelayed] = useState(false);
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setDelayed(true);
    }, 800);
    return () => clearTimeout(timeoutId);
  }, []);

  return (
    <>
      {delayed && (
        <div className={styles.question_wrapper}>
          <div className={styles.box_wrapper}>
            <img src='/images/9gram_logo.png' alt='ai bot' />
            <div className={styles.chat_wrapper}>
              <div className={styles.text_wrapper}>
                <div className={`${styles.text} r-regular`}>{text}</div>
                <div className={styles.button_wrapper}>
                  <>
                    {button.map((btn, idx) => (
                      <div key={`chatBtn-${idx}`}>
                        <ButtonCommon
                          variant='default-active'
                          size='med-small'
                          onClickBtn={(e) => {
                            handleOnClick(e, idx);
                            setClicked(true);
                          }}
                          disabled={disabled || clicked}
                          customClassName={
                            (disabled || clicked) && styles.disable_btn
                          }
                        >
                          {btn.text}
                        </ButtonCommon>
                      </div>
                    ))}
                  </>
                </div>
              </div>
              {toSave && (
                <div className={styles.scrap_wrapper} onClick={handleScrap}>
                  <Scrap scrap={scrap} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      {toast && (
        <MiniToast setToast={setToast} text='서랍에 답변이 저장되었습니다.' />
      )}
    </>
  );
};

export default BotBox;

import styles from '@components/pages/ai-analyze/minitoast.module.css';
import { useEffect } from 'react';

interface Props {
  setToast: Function;
  text: string;
}

const MiniToast = ({ setToast, text }: Props) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      setToast(false);
    }, 1500);
    return () => {
      clearTimeout(timer);
    };
  }, [setToast]);

  return <div className={`${styles.toast} s-regular`}>{text}</div>;
};

export default MiniToast;

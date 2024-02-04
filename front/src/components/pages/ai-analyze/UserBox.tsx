import styles from '@components/pages/ai-analyze/box.module.css';
import { useEffect, useState } from 'react';

interface Props {
  text: string;
}

const UserBox = ({ text }: Props) => {
  const [delayed, setDelayed] = useState(false);
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setDelayed(true);
    }, 700);
    return () => clearTimeout(timeoutId);
  }, []);

  return (
    <div className={styles.user_wrapper}>
      {delayed && <div className={`${styles.user_text} r-regular`}>{text}</div>}
    </div>
  );
};

export default UserBox;

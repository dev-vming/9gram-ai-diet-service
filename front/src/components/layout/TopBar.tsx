import Back from '@assets/Back';
import Drawer from '@assets/Drawer';
import Qicon from '@assets/Qicon';
import Toast from '@components/UI/Toast';
import ToastText from '@components/UI/ToastText';
import styles from '@components/layout/layout.module.css';
import { getKeyFromUrl } from '@utils/getNavProps';
import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { TopBarPropsType, TopNavKeyType } from 'typings/propTypes';

const TopBar = ({ home, title, back, qIcon, icon }: TopBarPropsType) => {
  const location = useLocation();
  const nowLocation = location.pathname.slice(1);
  const key: TopNavKeyType | string = getKeyFromUrl(nowLocation);
  const [showToast, setShowToast] = useState(false);
  const [position, setPosition] = useState({
    x: window.innerWidth / 2,
    y: window.innerHeight / 2,
  });
  const navigate = useNavigate();
  const onClickBack = () => {
    if (key === 'recordymdmeal') {
      return navigate(`/record/${nowLocation.split('/')[1]}`);
    }
    navigate(-1);
  };

  const onClickQicon = (e: React.MouseEvent<HTMLDivElement>) => {
    setShowToast(true);
    setPosition((prev) => ({ ...prev, x: e.clientX, y: e.clientY }));
  };

  if (key === 'aidrawerdetail') {
    const date = new URLSearchParams(window.location.search).get('date');
    const newDate = date?.split('-').join('.');
    title += `${newDate} 답변`;
  }

  return (
    <div className={styles.top}>
      {!home && (
        <div
          className={`${styles.back} ${back ? '' : styles['non-visible']}`}
          onClick={onClickBack}
        >
          <Back />
        </div>
      )}

      <div className={styles.title_wrapper}>
        <p className={home ? styles['name'] : styles.title}>{title}</p>
        <div className={qIcon ? styles.icon : styles['non-visible']}>
          <Qicon onClick={onClickQicon} />
        </div>
      </div>
      <Toast
        show={showToast}
        // position={position}
        setShow={setShowToast}
        toastPosCls={key === 'recordymdmeal' ? 'top' : 'middle'}
        time={50000}
      >
        <ToastText>
          {key === 'recordymdmeal' &&
            `사진을 올리고 AI로 간편하게 \n 식단을 기록해보세요!`}
          {key === 'addphotoymdmeal' &&
            '촬영 1회 안에 모든 음식이 담길 수 있도록 촬영해주세요.'}
        </ToastText>
      </Toast>

      <div
        className={`${styles.icon_wrapper} ${icon ? '' : styles['non-visible']}`}
        onClick={() => navigate(`/ai-drawer`)}
      >
        <Drawer />
      </div>
    </div>
  );
};

export default TopBar;

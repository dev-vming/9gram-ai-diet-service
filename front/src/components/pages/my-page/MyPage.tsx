import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import style from './mypage.module.css';
import { PencilIcon } from '@assets/PencilIcon';
import { mapGoaltoMsg, mapActivitytoMsg, findKeyByValue } from './mapMsg';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '@components/store';

const MyPage = () => {
  const userData = useSelector((state: RootState) => state.user.userInfo);
  const navigate = useNavigate();

  const goalMsg = mapGoaltoMsg[Number(userData?.dietGoal)];
  const activityMsg = mapActivitytoMsg[Number(userData?.activityAmount)];

  const handleIconClick = () => {
    navigate(`/my-page/edit`, {
      state: { userData, goalMsg, activityMsg },
    });
  };

  const mypageInfo = [
    {
      title: '신체 데이터',
      content: (
        <>
          {userData.height} cm <span style={{ color: 'black' }}>/</span>{' '}
          {userData.weight} kg
        </>
      ),
      titleClass: style.infoTitle,
      contentClass: style.infoContent,
    },
    {
      title: '활동량',
      content: activityMsg ? activityMsg : '활동량 설정',
      titleClass: style.infoTitle,
      contentClass: style.infoContent,
    },
  ];

  const handleSettingNavigate = () => {
    navigate('/my-page/settings', {
      state: { nickname: userData.username, gender: userData.gender },
    });
  };

  return (
    <>
      <div className={style.userProfileArea}>
        <div className={style.userProfileContainer}>
          {userData.profileImage ? (
            <img
              className={style.userProfile}
              src={userData.profileImage}
              alt='사용자 프로필'
            />
          ) : (
            <div className={style.defaultProfile}></div>
          )}

          <div className={style.userName}>{userData.username}</div>
        </div>
        <div className={style.editUserProfile} onClick={handleIconClick}>
          <PencilIcon width='15px' height='15px' strokeWidth={2} />
          <div className={style.editText}> 편집 </div>
        </div>
      </div>

      <div className={style.goalInfoArea}>
        <div className={style.goalInfoTitle}>
          <div className={style.infoTitle}>목표</div>
          <div className={style.infoContent}>
            {goalMsg ? `${goalMsg}` : '목표설정'}
          </div>
        </div>
        <div className={style.goaltInfo}>
          <div className={style.goalTitle}>목표 칼로리</div>
          <div className={style.goalDetail}>{userData.targetCalories}kcal</div>
        </div>
      </div>

      {mypageInfo.map((info, idx) => (
        <div key={idx} className={style.infoArea}>
          <div className={info.titleClass}>{info.title}</div>
          <div className={info.contentClass}>{info.content}</div>
        </div>
      ))}

      <div className={style.activityAccountArea}>
        <div className={style.infoTitle}>계정 설정</div>
        <img
          className={style.rightButton}
          src='/icons/right-arrow-icon.png'
          alt='계정 설정 화살표'
          onClick={handleSettingNavigate}
        />
      </div>
    </>
  );
};

export default MyPage;

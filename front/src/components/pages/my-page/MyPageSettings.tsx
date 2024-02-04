import { useLocation, useNavigate } from 'react-router-dom';
import style from './mypagesettings.module.css';
import { useEffect, useState } from 'react';
import useApi from '@hooks/useApi';
import { Modal } from '../../UI/Modal';
import { mapSelectModalMsg } from '../../UI/Modal';
import MyPageDropdown from './MyPageDropdwon';
import { findKeyByValue, gendertoMsg } from './mapMsg';
import { loginUser } from '@components/store/userLoginRouter';
import { useDispatch, useSelector } from 'react-redux';
import { logoutUser } from '@components/store/userLoginRouter';
import { RootState } from '@components/store';
import Toast from '@components/UI/Toast';
import ToastText from '@components/UI/ToastText';
import { log } from 'console';

const genderArr = ['남성', '여성', '기타'];

interface Result {
  config: {};
  data: string;
  headers: {};
  request: {};
  status: number;
  statusText: string;
}

const MyPageSettings = () => {
  const location = useLocation();
  const { nickname, gender } = location.state;
  const [isEditing, setisEditing] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalSelect, setModalSelect] = useState('');
  const [modalMsg, setModalMsg] = useState('');
  const [genderSelect, setGenderSelect] = useState(gendertoMsg[gender]);
  const [isGenderDropdwonVisible, setGenderDropdownVisible] = useState(false);
  const [newNickname, setNewNickName] = useState(nickname);
  const [toastText, setToastText] = useState('');
  const [showToast, setShowToast] = useState(false);
  const [logOutValid, setLogOutValid] = useState(false);
  const nickNameGet = useApi<Nickname>({});
  const nickNamePut = useApi<Nickname>({
    method: 'put',
  });
  const logOutGet = useApi<Result>({});
  const withdrawalGet = useApi<Result>({});

  const navigate = useNavigate();
  const dispatch = useDispatch();
  const userData = useSelector((state: RootState) => state.user.userInfo);

  type Nickname =
    | {
        data?: { isAvailable: boolean };
      }
    | undefined;

  const handleEditName = () => {
    if (!isEditing) {
      setisEditing(true);
    } else {
      nickNameGet.trigger({
        path: `/user/username/${newNickname}`,
        method: 'get',
      });
    }
  };

  const handleToast = () => {
    setShowToast(true);
  };

  useEffect(() => {
    if (nickNameGet.result?.data?.isAvailable) {
      const updateData = {
        ...userData,
        username: newNickname,
      };
      setNewNickName(newNickname);
      nickNamePut.trigger({
        path: '/user',
        data: { username: newNickname },
      });
      dispatch(loginUser(updateData));
      setisEditing(false);
    } else if (nickNameGet.result?.data?.isAvailable === false) {
      setToastText('이미 존재하는 닉네임 입니다');
      handleToast();
    }
  }, [nickNameGet.result?.data]);

  const editNickName = () => {
    setisEditing(!isEditing);
  };

  const handleSelect = (value: string) => {
    const newGenderValue = Number(findKeyByValue(gendertoMsg, value));
    if (newGenderValue) {
      const updateData = {
        ...userData,
        gender: newGenderValue,
      };
      setGenderSelect(gendertoMsg[newGenderValue]);
      dispatch(loginUser(updateData));
      nickNameGet.trigger({
        path: '/user',
        method: 'put',
        data: { gender: newGenderValue },
      });
    }
    setGenderDropdownVisible(false);
  };

  useEffect(() => {
    if (userData.gender) {
      setGenderSelect(gendertoMsg[Number(userData.gender)]);
    }
  }, [genderSelect, userData.gender]);

  const handleLogOut = () => {
    setShowModal(true);
    setModalSelect('login');
    setModalMsg(mapSelectModalMsg.login);
  };

  const handleConfirmLogout = () => {
    logOutGet.trigger({ path: '/auth/logout' });
  };

  useEffect(() => {
    if (userData.username === '') {
      navigate('/');
    }
  }, [userData.username, logOutGet.result?.status]);

  useEffect(() => {
    if (
      withdrawalGet.result?.status === 200 ||
      logOutGet.result?.status === 200
    ) {
      dispatch(logoutUser());
      setShowModal(false);
      navigate('/');
    } else if (!logOutGet.error) {
      setToastText('로그아웃에 실패했습니다');
      handleToast();
    } else if (!withdrawalGet.error) {
      setToastText('회원 탈퇴에 실패했습니다');
      handleToast();
    }
  }, [logOutGet.result, withdrawalGet.result]);

  const handleWithdrawal = () => {
    setShowModal(true);
    setModalSelect('withdrawal');
    setModalMsg(mapSelectModalMsg.withdrawal);
  };

  const handleConfirmWithdrawal = () => {
    withdrawalGet.trigger({ path: '/auth/withdrawal' });
  };

  const confirmHandler =
    modalSelect === 'login' ? handleConfirmLogout : handleConfirmWithdrawal;

  return (
    <>
      {showModal && (
        <Modal
          modalSelect={modalSelect}
          modalMsg={modalMsg}
          onClose={() => setShowModal(false)}
          onConfirm={confirmHandler}
        />
      )}
      <div className={style.container}>
        <Toast show={showToast} setShow={setShowToast}>
          <ToastText>{toastText}</ToastText>
        </Toast>
        <div className={style.accountWrapper}>
          <div className={style.detailContainer}>
            <div className={style.accountSettings}> 닉네임 </div>
            <div className={style.editContainer}>
              {isEditing ? (
                <>
                  <input
                    className={style.inputNickName}
                    placeholder={newNickname}
                    onChange={(e) => setNewNickName(e.target.value)}
                  />
                  <button
                    className={style.editButtonClicked}
                    onClick={handleEditName}
                  >
                    완료
                  </button>
                </>
              ) : (
                <>
                  <div className={style.showNickName}>{newNickname}</div>
                  <button className={style.editButton} onClick={editNickName}>
                    편집
                  </button>
                </>
              )}
            </div>
          </div>

          <div className={style.detailContainer}>
            <div className={style.accountSettings}> 성별 </div>
            <div className={style.selectGender}>
              <MyPageDropdown
                items={genderArr}
                selectedItem={genderSelect}
                onSelectItem={handleSelect}
                toggleDropdown={() =>
                  setGenderDropdownVisible(!isGenderDropdwonVisible)
                }
                isDropdownVisible={isGenderDropdwonVisible}
              />
            </div>
          </div>
          <div className={style.detailContainer}>
            <div className={style.accountSettings}> 등급 </div>
            <div className={style.accountFontStyle}> [유료] 새싹밀 </div>
          </div>
        </div>
        <div className={style.detailContainer}>
          <div className={style.accountSettings}> 앱 버전 </div>
          <div className={style.accountFontStyle}> 최신 버전 </div>
        </div>
        <div className={style.detailContainer}>
          <div className={style.accountSettings}> 로그아웃 </div>
          <div>
            <img
              className={style.rightButton}
              onClick={handleLogOut}
              src='/icons/right-arrow-icon.png'
              alt='계정 설정 화살표'
            />
          </div>
        </div>

        <div className={style.detailContainer}>
          <div className={style.accountSettings}> 탈퇴하기 </div>
          <div>
            <img
              className={style.rightButton}
              onClick={handleWithdrawal}
              src='/icons/right-arrow-icon.png'
              alt='계정 설정 화살표'
            />
          </div>
        </div>
      </div>
    </>
  );
};

export default MyPageSettings;

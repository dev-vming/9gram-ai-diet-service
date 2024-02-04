import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ButtonCommon from '@components/UI/ButtonCommon';
import InputCommon from '@components/UI/InputCommon';
import './Login.css';
import { useDispatch, useSelector } from 'react-redux';
import { UserInfo, loginUser } from '@components/store/userLoginRouter';
import useCachingApi from '@hooks/useCachingApi';
import Toast from '@components/UI/Toast';
import ToastText from '@components/UI/ToastText';
import { checkValuesNullOrEmpty } from '@utils/checkValuesNullOrEmpty';

const Login = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [toastText, setToastText] = useState('');
  const [showToast, setShowToast] = useState(false);
  const {
    loading,
    trigger: triggerLogin,
    result: postResult,
    error,
  } = useCachingApi<any>({
    method: 'post',
    path: '/auth/local/login',
  });
  const { trigger: triggerUserInfo } = useCachingApi<any>({
    method: 'get',
    path: '/user',
    //gcTime: 24 * 60 * 60 * 1000, //쿠키 만료시간
  });

  const userInfo = useSelector<{
    user: {
      userInfo: UserInfo;
    };
  }>((state) => state.user?.userInfo?.username);
  const healthInfo = useSelector<{
    user: {
      userInfo: UserInfo;
    };
  }>((state) => state.user?.userInfo?.height);

  const handleEmailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEmail(e.target.value);
  };

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPassword(e.target.value);
  };
  const handleToast = () => {
    setShowToast(true);
  };

  const handleLogin: (
    event: React.MouseEvent<HTMLButtonElement, MouseEvent>
  ) => void = async () => {
    triggerLogin(
      { email, password },
      {
        onSuccess: (data) => {
          if (
            data ===
            '등록되지 않은 이메일 이거나, 유효하지 않은 비밀번호입니다.'
          ) {
            setToastText(data);
            handleToast();
          } else {
            dispatch(loginUser(data.data));
          }
          if (checkValuesNullOrEmpty(data)) {
          }
        },
      }
    );
  };

  useEffect(() => {
    userInfo && healthInfo && navigate('/home');
  }, [userInfo]);

  useEffect(() => {
    //로그인 화면 들어갔을때 쿠키 확인 절차
    triggerUserInfo('', {
      onSuccess: (data) => {
        dispatch(loginUser(data.data));
      },
    });
  }, []);

  useEffect(() => {
    //있으면 home
    console.log(postResult);
    if (
      postResult &&
      postResult.status === 200 &&
      postResult.data !== '로그인이 필요합니다.'
    ) {
      if (
        healthInfo === '' ||
        healthInfo === undefined ||
        healthInfo === 0 ||
        healthInfo === null
      ) {
        navigate('/onboarding/1');
      } else {
        navigate('/home');
      }
    }
  }, [postResult]);

  return (
    <div className='login-container'>
      <Toast show={showToast} setShow={setShowToast}>
        <ToastText>{toastText}</ToastText>
      </Toast>
      <div className='body'>
        <p className='r-large'>이메일</p>
      </div>
      <div style={{ marginTop: '15px' }}>
        <InputCommon
          className={`button large r-large gray`}
          value={email}
          onChange={handleEmailChange}
        />
      </div>
      <div className='body'>
        <p className='r-large'>비밀번호</p>
      </div>
      <div style={{ marginTop: '15px' }}>
        <InputCommon
          className={`button large r-large gray`}
          type='password'
          value={password}
          onChange={handlePasswordChange}
        />
      </div>
      <div className='button-container'>
        <div style={{ marginBottom: '20px' }}>
          <p className='r-medium'>비밀번호를 잊으셨나요?</p>
        </div>
        <ButtonCommon
          variant='default-active'
          size='big'
          onClickBtn={handleLogin}
          // disabled={loading}
        >
          {/* {loading ? '로그인 하는중' : '로그인'} */}
          {'로그인'}
        </ButtonCommon>
      </div>
    </div>
  );
};

export default Login;

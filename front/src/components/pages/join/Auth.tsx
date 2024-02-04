import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ButtonCommon from '@components/UI/ButtonCommon';
import './Auth.css';
import logo from './9g_logo.png';
import { useSelector } from 'react-redux';
import { RootState } from '@components/store';

const Auth = () => {
  const navigate = useNavigate();
  const userInfo = useSelector(
    (state: RootState) => state.user?.userInfo?.username
  );
  const handleClickGoogleAuth = () => {
    window.location.href = 'http://localhost:5001/api/auth/google/login';
  };

  // useEffect(() => {
  //   if (userInfo) {
  //     navigate('/home');
  //   }
  // }, [userInfo]);

  return (
    <div className='center-container'>
      <div className='logo-container'>
        <h1 className='b-regular'>9g</h1>
        <h1 className='b-big'>구그램</h1>
        <img src={logo} className='logo' />
        <p className='s-big'> 바쁜 당신을 위한</p>
        <p className='s-superlarge'>AI 영양사</p>
        <div style={{ marginTop: '60px' }}>
          <ButtonCommon
            variant='default-active'
            size='big'
            onClickBtn={() => navigate(`/login`)}
          >
            로그인
          </ButtonCommon>
        </div>
        <div style={{ marginTop: '10px' }}>
          <ButtonCommon
            customClassName='white'
            size='big'
            onClickBtn={() => navigate(`/join`)}
          >
            회원가입
          </ButtonCommon>
        </div>
        <div style={{ marginTop: '10px' }}>
          <ButtonCommon
            customClassName='lightblue'
            size='big'
            onClickBtn={handleClickGoogleAuth}
          >
            Google로 계속하기
          </ButtonCommon>
        </div>
      </div>
    </div>
  );
};

export default Auth;

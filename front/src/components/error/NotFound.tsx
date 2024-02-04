import { Link } from 'react-router-dom';

const NotFound = () => {
  return (
    <div className='error__wrapper'>
      <div className='error'>
        <div className='error__head'>원하시는 페이지를 찾을 수 없습니다</div>
        <div className='error__content'>
          요청하신 페이지의 주소가 잘못 입력되었거나,
          <br />
          주소의 변경 혹은 삭제로 인해 사용하실 수 없습니다.
          <br />
          입력하신 페이지의 주소가 정확한지 다시 한 번 확인해주세요.
        </div>
      </div>
      <Link to='/home' className='error__button'>
        홈으로 돌아가기
      </Link>
    </div>
  );
};

export default NotFound;

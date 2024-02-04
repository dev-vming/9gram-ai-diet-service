import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Modal } from '../../UI/Modal';
const GoogleAuth = () => {
  const navigate = useNavigate();
  const [_, setShowModal] = useState(false);
  const onConfirm = () => {
    navigate('/home');
  };

  return (
    <Modal
      onClose={() => setShowModal(false)}
      onConfirm={onConfirm}
      modalMsg='구글 인증에 성공하였습니다!'
      modalSelect='홈으로 가기'
    />
  );
};

export default GoogleAuth;

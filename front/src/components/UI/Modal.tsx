import { useEffect } from 'react';
import style from './modal.module.css';

interface ModalProps {
  onClose: () => void;
  onConfirm: () => void;
  modalSelect: string;
  modalMsg: string;
}

export const mapSelectModalMsg = {
  login: '로그아웃 하시겠습니까?',
  withdrawal: '정말로 탈퇴하시게요..? ( Ĭ ^ Ĭ )',
  mealDelete: '기록을 삭제하겠습니까 (  ˙ỏ˙ ) ？',
};

export const Modal = ({
  onClose,
  onConfirm,
  modalMsg,
  modalSelect,
}: ModalProps) => {
  useEffect(() => {}, [modalSelect]);
  return (
    <div className={style.modal}>
      <div className={style.modalContent}>
        <h2>{modalMsg}</h2>
        <div>
          <button className={style.buttonYes} onClick={onConfirm}>
            확인
          </button>
          <button className={style.buttonNo} onClick={onClose}>
            취소
          </button>
        </div>
      </div>
    </div>
  );
};

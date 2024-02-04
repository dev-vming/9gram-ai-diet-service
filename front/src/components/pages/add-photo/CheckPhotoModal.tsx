import ButtonCommon from '@components/UI/ButtonCommon';
import styles from './checkphotomodal.module.css';
import React, { useRef } from 'react';
import { api2 } from '@utils/axiosConfig';
import { useNavigate, useParams } from 'react-router-dom';

interface Props {
  pre: string;
  imgUrl: string;
  setShowModal: React.Dispatch<React.SetStateAction<boolean>>;
}

const fileName = new Date().getTime() + Math.random().toString().split(".")[0];

const CheckPhotoModal = ({ pre, imgUrl, setShowModal }: Props) => {
  const formData = new FormData();
  const cropRef = useRef<HTMLCanvasElement | null>(null);
  const params = useParams();
  const date = params.date;
  const mealTime = params.mealTime;
  const navigate = useNavigate();
  
  function base64toFile(base_data: string, filename: string) {
    const arr = base_data.split(',');
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);

    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }

    return new File([u8arr], filename, { type: 'image/jpg' });
  }

  let canvasUrl:string|undefined;
  const uploadImg = () => {
    const canvas = cropRef.current;
    const context = canvas?.getContext('2d');

    const image = new Image();
    image.src = imgUrl;

    const canvasW = 350;
    const canvasH = 200;
    const scale = Math.max(canvasW / image.width, canvasH / image.height);
    const offsetX = (canvasW - image.width * scale) / 2;
    const offsetY = (canvasH - image.height * scale) / 2;

    context?.drawImage(
      image,
      offsetX,
      offsetY,
      image.width * scale,
      image.height * scale
    );

    canvasUrl = canvas?.toDataURL();
    const file = base64toFile(canvasUrl as string, fileName);
    formData.append('image', file);
    postFile();
  };

  const postFile = async () => {
    const res = await api2.post('/classification', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    const foodsArr = res.data.map((food: any) => ({ ...food, counts: 1}))

    const data = {
      imgUrl: canvasUrl,
      foods: foodsArr,
    };
    
    navigate(`/record/${date}/${mealTime}/edit`, { state: data })
  };
  

  

  return (
    <div className={styles.container}>
      <div className={styles.modalbox}>
        <div className={styles.title}>{pre} 사진 확인</div>
        <div className={styles.contentbox}>
          <p className='r-large' style={{ marginBottom: '20px' }}>
            AI가 분석할 사진을 확인해주세요!
          </p>
          <div style={{ position: 'relative' }}>
            <img className={styles.photo} src={imgUrl} alt='촬영한사진' />
            <canvas
              width='350'
              height='200'
              style={{ position: 'absolute', bottom: '0', display:'none'}}
              ref={cropRef}
            />
          </div>
        </div>
        <div className={styles.btnbox}>
          <ButtonCommon
            size='medium'
            variant='disabled'
            onClick={() => setShowModal(false)}
          >
            재{pre}
          </ButtonCommon>
          <ButtonCommon
            size='medium'
            variant='default-active'
            onClick={uploadImg}
          >
            사진 분석
          </ButtonCommon>
        </div>
      </div>
    </div>
  );
};

export default CheckPhotoModal;

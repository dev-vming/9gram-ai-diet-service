import { useNavigate, useParams } from 'react-router-dom';
import styles from './addphoto.module.css';
import { useEffect, useRef, useState } from 'react';
import CheckPhotoModal from './CheckPhotoModal';

const AddPhoto = () => {
  const navigate = useNavigate();
  const params = useParams();
  const date = params.date;
  const mealTime = params.mealTime;

  const selectFile = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [showModal, setShowModal] = useState(false);
  const [imgUrl, setImgUrl] = useState('');
  const [pre, setPre] = useState('선택');

  useEffect(() => {
    getWebcam((stream: MediaProvider) => {
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      streamRef.current = stream as MediaStream;
    });
    return () => {
      if (!streamRef.current) return;
      streamRef.current.getTracks().forEach((track) => {
        streamRef.current?.removeTrack(track);
        track.stop();
      });
    };
  }, []);

  const userAgent = navigator.userAgent;
  const isMobile = userAgent.match(/iPhone/i) || userAgent.match(/Android/i);

  const getWebcam = (callback: (stream: MediaProvider) => void) => {
    try {
      const constraints = isMobile
        ? {
            video: {
              facingMode: { exact: 'environment' },
              width: 390,
              height: 550,
            },
            audio: false,
          }
        : {
            video: {
              width: 390,
              height: 550,
            },
            audio: false,
          };
      navigator.mediaDevices.getUserMedia(constraints).then(callback);
    } catch (err) {
      console.log(err);
      return undefined;
    }
  };

  const screenShot = () => {
    const video = document.getElementById('videoCam');
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!video) return;
    const cropX = (video?.offsetWidth - 350) / 2;
    const cropY = (video?.offsetHeight - 200) / 2;

    context?.drawImage(
      video as CanvasImageSource,
      cropX,
      cropY,
      350,
      200,
      0,
      0,
      350,
      200
    );

    const image = canvas?.toDataURL();
    image && setImgUrl(image);
    setPre('촬영');
    setShowModal(true);
  };

  const checkImg = () => {
    if (!selectFile.current) return;
    if (!selectFile.current.files) return;
    const file = selectFile.current.files[0];

    const reader = new FileReader();
    reader.onload = (e) => {
      setImgUrl(e.target?.result as string);
    };
    reader.readAsDataURL(file);
    setPre('선택');
    setShowModal(true);
  };

  return (
    <div style={{ position: 'relative' }}>
      <div className={styles.cambox}>
        <div className={styles.guide}></div>
        <canvas
          width='350'
          height='200'
          style={{ display: 'none', position: 'absolute' }}
          ref={canvasRef}
        />
        <video
          id='videoCam'
          className={styles.cam}
          ref={videoRef}
          autoPlay
          playsInline
        />
      </div>
      <div className={styles.addbox}>
        <input
          type='file'
          accept='image/*'
          style={{ display: 'none' }}
          ref={selectFile}
          onChange={checkImg}
        />
        <div
          className={styles.item}
          onClick={() => selectFile.current?.click()}
        >
          <img className={styles.icon} src='/icons/album.png' alt='앨범' />
          <div className={styles.text}>앨범</div>
        </div>
        <button className={styles.shotbtn} onClick={screenShot}></button>
        <div
          className={styles.item}
          onClick={() => navigate(`/add-photo/${date}/${mealTime}/search`)}
        >
          <img className={styles.icon} src='/icons/self.png' alt='직접입력' />
          <div className={styles.text}>직접입력</div>
        </div>
      </div>
      {showModal && (
        <CheckPhotoModal
          pre={pre}
          imgUrl={imgUrl}
          setShowModal={setShowModal}
        />
      )}
    </div>
  );
};

export default AddPhoto;

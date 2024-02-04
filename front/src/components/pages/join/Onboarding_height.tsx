import { useRef } from 'react';
import './Onboarding.css';
import NumericPad from './NumericPad';

const Onboarding_height = () => {
  const inputHeight = useRef(null);

  return (
    <div>
      <h1 className='b-medium'>당신의 키는 얼마인가요?</h1>
      <div style={{ marginTop: '130px', textAlign: 'center' }}>
        <input
          type='text'
          ref={inputHeight}
          placeholder='190'
          className='b-super'
          style={{
            width: '150px',
            textAlign: 'center',
            display: 'inline-block',
            border: 'none',
          }}
        />
        <span
          style={{ marginLeft: '5px', verticalAlign: 'top' }}
          className='b-super'
        >
          cm
        </span>
      </div>
      <div style={{ marginTop: '50px' }}>
        <NumericPad inputRef={inputHeight} />
      </div>
    </div>
  );
};

export default Onboarding_height;

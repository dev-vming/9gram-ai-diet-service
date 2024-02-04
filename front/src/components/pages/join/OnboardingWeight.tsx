import { useEffect, useState } from 'react';
import './Onboarding.css';
import NumericPad from './NumericPad';
import { OnboardingProps } from './OnboardingGender';

const OnboardingWeight: React.FC<OnboardingProps> = ({
  userData,
  onClickOnboarding,
}) => {
  const [inputValue, setInputValue] = useState(
    userData.weight?.toString() || ''
  );

  useEffect(() => onClickOnboarding({ ['weight']: inputValue }), [inputValue]);
  return (
    <div>
      <h1 className='b-medium'>당신의 몸무게는 얼마인가요?</h1>
      <div style={{ marginTop: '130px', textAlign: 'center' }}>
        <input
          type='text'
          placeholder='100'
          className='b-super'
          style={{
            width: '150px',
            textAlign: 'center',
            display: 'inline-block',
            border: 'none',
          }}
          value={inputValue}
          onChange={() => {}}
        />
        <span
          style={{ marginLeft: '5px', verticalAlign: 'top' }}
          className='b-super'
        >
          kg
        </span>
      </div>
      <div style={{ marginTop: '50px' }}>
        <NumericPad setInputValue={setInputValue} />
      </div>
    </div>
  );
};

export default OnboardingWeight;
